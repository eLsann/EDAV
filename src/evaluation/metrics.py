import numpy as np

def calculate_metrics(similarities, labels, thresholds=None):
    """
    Menghitung FAR, FRR, dan Akurasi dari prediksi model.
    Args:
        similarities: list of float, hasil cosine similarity
        labels: list of int, 1 untuk SAMA, 0 untuk BEDA
        thresholds: array/list threshold untuk diuji. Jika None, gunakan 0.0 s.d 1.0.
    """
    y_scores = np.array(similarities)
    y_true = np.array(labels)
    
    if thresholds is None:
        thresholds = np.arange(-1.0, 1.01, 0.01)
        
    best_acc = 0.0
    best_thresh = 0.0
    best_far = 0.0
    best_frr = 0.0
    
    metrics_log = []
    
    for thresh in thresholds:
        # Prediksi bernilai 1 jika similarity > thresh, else 0
        y_pred = (y_scores >= thresh).astype(int)
        
        # True Positives, False Positives, dll
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        # Total SAMA dan BEDA di dataset (P = Positives, N = Negatives)
        p = tp + fn
        n = tn + fp
        
        # Menghindari division by zero
        far = fp / n if n > 0 else 0.0
        frr = fn / p if p > 0 else 0.0
        acc = (tp + tn) / (p + n) if (p + n) > 0 else 0.0
        
        metrics_log.append({
            'threshold': thresh,
            'far': far,
            'frr': frr,
            'acc': acc
        })
        
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            best_far = far
            best_frr = frr
            
    return {
        'best_acc': best_acc,
        'best_thresh': best_thresh,
        'best_far': best_far,
        'best_frr': best_frr,
        'all_metrics': metrics_log
    }

def find_best_threshold(similarities, labels, thresholds=None):
    """
    Hanya mencari threshold terbaik dari kumpulan data development.
    (Mencegah optimasi dari himpunan test).
    Returns threshold yang memberikan Accuracy tertinggi.
    """
    res = calculate_metrics(similarities, labels, thresholds)
    return res['best_thresh']

def evaluate_with_threshold(similarities, labels, threshold):
    """
    Mengevaluasi secara pasif himpunan data uji (Test Set) menggunakan threshold baku.
    Tidak ada pencarian atau iterasi threshold di fungsi ini.
    """
    y_scores = np.array(similarities)
    y_true = np.array(labels)
    
    y_pred = (y_scores >= threshold).astype(int)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    p = tp + fn
    n = tn + fp
    
    far = fp / n if n > 0 else 0.0
    frr = fn / p if p > 0 else 0.0
    acc = (tp + tn) / (p + n) if (p + n) > 0 else 0.0
    
    return {
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'far': float(far),
        'frr': float(frr),
        'acc': float(acc),
        'threshold_used': float(threshold)
    }

