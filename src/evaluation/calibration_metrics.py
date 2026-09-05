import numpy as np

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Menghitung Expected Calibration Error (ECE).
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    for i in range(n_bins):
        bin_idx = binids == i
        if np.sum(bin_idx) > 0:
            prob_mean = np.mean(y_prob[bin_idx])
            acc_mean = np.mean(y_true[bin_idx])
            
            ece += (np.sum(bin_idx) / len(y_prob)) * np.abs(prob_mean - acc_mean)
            
    return ece

def brier_score(y_true, y_prob):
    """
    Menghitung Brier Score (Mean Squared Error dari Probabilitas prediksi terhadap label asli).
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    return np.mean((y_prob - y_true)**2)
