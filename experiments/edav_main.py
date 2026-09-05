import os
import sys
import torch
import numpy as np
import json
import csv
from collections import defaultdict

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.dataset.ytf_loader import YTFDataLoader
from src.similarity.cosine import compute_pairwise_cosine
from src.calibration.logistic import ScoreCalibrator
from src.detection.mtcnn_detector import FaceDetector
from src.sampling.deterministic import sample_frames_equidistant

detector = None

def get_temporal_mask(video_name, loader, num_frames=5):
    global detector
    if detector is None:
        detector = FaceDetector()
    all_frames = loader.get_all_frame_paths(video_name)
    sampled = sample_frames_equidistant(all_frames, num_frames)
    mask = []
    for f in sampled:
        tensor, prob, _ = detector.detect_and_crop(f)
        mask.append(tensor is not None)
    return mask

def load_sequential_embeddings(video_name, t, loader):
    safe_vname = video_name.replace('/', '_').replace('\\', '_')
    cache_path = os.path.join(root_path, 'cache', 'embeddings_N5', f"{safe_vname}.pt")
    if not os.path.exists(cache_path): return [], False
    emb = torch.load(cache_path)
    if emb is None: return [], False
    
    total_valid = len(emb)
    if total_valid == 5:
        return emb[:t], True
    else:
        mask = get_temporal_mask(video_name, loader, 5)
        successes_up_to_t = sum(mask[:t])
        current_valid = mask[t-1]
        if successes_up_to_t == 0:
            return [], current_valid
        return emb[:successes_up_to_t], current_valid

def get_conf_bin(p):
    if p is None: return -1
    if p < 0.1: return 0
    if p < 0.3: return 1
    if p < 0.7: return 2
    if p < 0.9: return 3
    return 4

def run_blind_test():
    print("==================================================")
    print(" FULL 10-FOLD BLIND TEST OF EDAV")
    print("==================================================")
    
    mat_path = os.path.join(root_path, 'dataset ytf', 'meta_data', 'meta_and_splits.mat')
    frames_dir = os.path.join(root_path, 'dataset ytf', 'frame_images_DB')
    loader = YTFDataLoader(mat_path, frames_dir)
    
    C_FA = 15.0
    C_FR = 1.0
    C_OBS = 0.1
    
    out_dir = os.path.join(root_path, 'results', 'edav_final')
    os.makedirs(out_dir, exist_ok=True)
    
    print("1. Pre-computing Sequential Cosine Similarities for all 10 folds...")
    # fold_data[fold][i][t] = {'sim': float|None, 'valid': bool, 'y': int, 'v1': str, 'v2': str}
    fold_data = {f: [] for f in range(1, 11)}
    
    for fold in range(1, 11):
        pairs = loader.get_fold(fold)
        print(f"   Processing Fold {fold} ({len(pairs)} pairs)...")
        for i, pair in enumerate(pairs):
            traj = {'y': pair['label'], 'v1': pair['video1'], 'v2': pair['video2']}
            for t in range(1, 6):
                e1, v1_val = load_sequential_embeddings(pair['video1'], t, loader)
                e2, v2_val = load_sequential_embeddings(pair['video2'], t, loader)
                valid = v1_val and v2_val
                sim = None
                if len(e1) > 0 and len(e2) > 0:
                    sim = compute_pairwise_cosine(e1, e2)
                traj[t] = {'sim': sim, 'valid': valid}
            fold_data[fold].append(traj)

    print("\n2. Executing 10-Fold Leave-One-Fold-Out Blind Test...")
    
    results = {'EDAV': [], 'Fixed-1': [], 'Fixed-3': [], 'Fixed-5': []}
    
    for test_fold in range(1, 11):
        print(f"   >>> TARGET TEST FOLD {test_fold} <<<")
        
        # Collect Dev Data
        dev_trajectories = []
        for dev_fold in range(1, 11):
            if dev_fold != test_fold:
                dev_trajectories.extend(fold_data[dev_fold])
                
        # Fit Calibrators
        calibrators = {}
        for t in range(1, 6):
            sims = []
            lbls = []
            for traj in dev_trajectories:
                if traj[t]['sim'] is not None:
                    sims.append(traj[t]['sim'])
                    lbls.append(traj['y'])
            c = ScoreCalibrator()
            c.fit(sims, lbls)
            calibrators[t] = c
            
        # Assign p_I and Backward Induction
        V = [{} for _ in range(len(dev_trajectories))]
        Q_continue = {}
        
        for i, traj in enumerate(dev_trajectories):
            for t in range(1, 6):
                sim = traj[t]['sim']
                if sim is not None:
                    traj[f'p_I_{t}'] = calibrators[t].predict_proba([sim])[0]
                else:
                    traj[f'p_I_{t}'] = None
                    
        for i, traj in enumerate(dev_trajectories):
            p = traj['p_I_5']
            if p is not None:
                V[i][5] = min(C_FA * (1 - p), C_FR * p)
            else:
                V[i][5] = min(C_FA, C_FR)
                
        for t in range(4, 0, -1):
            bins = {}
            for i, traj in enumerate(dev_trajectories):
                p = traj[f'p_I_{t}']
                valid = traj[t]['valid']
                s = (t, valid, get_conf_bin(p))
                if s not in bins: bins[s] = []
                bins[s].append(V[i][t+1] + C_OBS)
                
            for s, vals in bins.items():
                Q_continue[s] = np.mean(vals)
                
            for i, traj in enumerate(dev_trajectories):
                p = traj[f'p_I_{t}']
                valid = traj[t]['valid']
                s = (t, valid, get_conf_bin(p))
                L_stop = min(C_FA * (1 - p), C_FR * p) if p is not None else min(C_FA, C_FR)
                V[i][t] = min(L_stop, Q_continue[s])
                
        # SAVE DEV PARAMETERS FOR AUDIT
        fold_out_dir = os.path.join(out_dir, f'fold_{test_fold}')
        os.makedirs(fold_out_dir, exist_ok=True)
        q_out = {str(k): v for k, v in Q_continue.items()}
        with open(os.path.join(fold_out_dir, 'risk_table.json'), 'w') as f:
            json.dump(q_out, f, indent=4)
            
        # ----------------------------------------------------
        # TEST PHASE ON BLIND FOLD
        # ----------------------------------------------------
        test_trajectories = fold_data[test_fold]
        
        methods = {
            'EDAV': {'t_stops': [], 'corr': [], 'fp': [], 'fn': [], 'stops': {1:0, 2:0, 3:0, 4:0, 5:0}},
            'Fixed-1': {'t_stops': [], 'corr': [], 'fp': [], 'fn': []},
            'Fixed-3': {'t_stops': [], 'corr': [], 'fp': [], 'fn': []},
            'Fixed-5': {'t_stops': [], 'corr': [], 'fp': [], 'fn': []}
        }
        
        for traj in test_trajectories:
            y = traj['y']
            
            # Helper to make cost-based decision given t
            def get_fixed_decision(t_fix):
                sim = traj[t_fix]['sim']
                if sim is None:
                    return 1 if C_FR > C_FA else 0
                p = calibrators[t_fix].predict_proba([sim])[0]
                return 1 if (C_FA * (1-p) < C_FR * p) else 0

            # 1. Evaluate Fixed baselines
            for f_t in [1, 3, 5]:
                m = f'Fixed-{f_t}'
                dec = get_fixed_decision(f_t)
                methods[m]['t_stops'].append(f_t)
                methods[m]['corr'].append(dec == y)
                if dec == 1 and y == 0: methods[m]['fp'].append(1)
                if dec == 0 and y == 1: methods[m]['fn'].append(1)
                
            # 2. Evaluate EDAV
            final_dec = None
            t_stop = 5
            for t in range(1, 6):
                sim = traj[t]['sim']
                valid = traj[t]['valid']
                p = calibrators[t].predict_proba([sim])[0] if sim is not None else None
                state = (t, valid, get_conf_bin(p))
                
                L_stop = min(C_FA * (1 - p), C_FR * p) if p is not None else min(C_FA, C_FR)
                
                if t == 5:
                    action = 'STOP'
                else:
                    q_cont = Q_continue.get(state, float('inf'))
                    action = 'NEXT' if q_cont < L_stop else 'STOP'
                    
                if action == 'STOP':
                    t_stop = t
                    if p is None:
                        final_dec = 1 if C_FR > C_FA else 0
                    else:
                        final_dec = 1 if (C_FA * (1-p) < C_FR * p) else 0
                    break
                    
            methods['EDAV']['t_stops'].append(t_stop)
            methods['EDAV']['stops'][t_stop] += 1
            methods['EDAV']['corr'].append(final_dec == y)
            if final_dec == 1 and y == 0: methods['EDAV']['fp'].append(1)
            if final_dec == 0 and y == 1: methods['EDAV']['fn'].append(1)

        # Aggregate fold metrics
        for m_name, m_data in methods.items():
            total_pos = sum(1 for traj in test_trajectories if traj['y'] == 1)
            total_neg = sum(1 for traj in test_trajectories if traj['y'] == 0)
            fp_c = sum(m_data['fp'])
            fn_c = sum(m_data['fn'])
            acc = np.mean(m_data['corr'])
            far = fp_c / total_neg if total_neg > 0 else 0
            frr = fn_c / total_pos if total_pos > 0 else 0
            avg_d = np.mean(m_data['t_stops'])
            
            row = {
                'fold': test_fold, 'method': m_name,
                'TP': total_pos - fn_c, 'TN': total_neg - fp_c,
                'FP': fp_c, 'FN': fn_c,
                'acc': acc, 'far': far, 'frr': frr, 'avg_depth': avg_d
            }
            if m_name == 'EDAV':
                for t in range(1, 6):
                    row[f'stop_t{t}'] = m_data['stops'][t] / len(test_trajectories)
                    
            results[m_name].append(row)
            
    print("\n3. Cross-Fold Summary")
    summary = []
    
    # Calculate means and stds
    for m in ['Fixed-1', 'Fixed-3', 'Fixed-5', 'EDAV']:
        recs = results[m]
        m_acc, s_acc = np.mean([r['acc'] for r in recs]), np.std([r['acc'] for r in recs])
        m_far, s_far = np.mean([r['far'] for r in recs]), np.std([r['far'] for r in recs])
        m_frr, s_frr = np.mean([r['frr'] for r in recs]), np.std([r['frr'] for r in recs])
        m_dep, s_dep = np.mean([r['avg_depth'] for r in recs]), np.std([r['avg_depth'] for r in recs])
        
        summary.append({
            'Method': m,
            'Mean Accuracy': m_acc, 'Std Accuracy': s_acc,
            'Mean FAR': m_far, 'Std FAR': s_far,
            'Mean FRR': m_frr, 'Std FRR': s_frr,
            'Mean Depth': m_dep, 'Std Depth': s_dep
        })
        
        print(f"[{m}] Acc: {m_acc*100:.2f}±{s_acc*100:.2f}% | FAR: {m_far*100:.2f}±{s_far*100:.2f}% | FRR: {m_frr*100:.2f}±{s_frr*100:.2f}% | Depth: {m_dep:.2f}±{s_dep:.2f}")

    # Write summary CSV
    with open(os.path.join(out_dir, 'blind_test_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)
        
    # Write EDAV details
    with open(os.path.join(out_dir, 'edav_fold_details.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results['EDAV'][0].keys())
        w.writeheader()
        w.writerows(results['EDAV'])

    print(f"\n[OK] Evaluation complete. Results frozen in {out_dir}")

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    run_blind_test()
