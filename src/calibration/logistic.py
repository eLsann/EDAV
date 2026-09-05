import numpy as np
from sklearn.linear_model import LogisticRegression

class ScoreCalibrator:
    def __init__(self):
        """
        Calibrator menggunakan Logistic Regression (Platt Scaling)
        untuk mengonversi cosine similarity menjadi probability p_I in [0, 1].
        """
        self.calibrator = LogisticRegression(solver='lbfgs')
        self.is_fitted = False
        
    def fit(self, similarities, labels):
        """
        Melatih calibrator berdasarkan data fold development.
        Args:
            similarities: array-like of shape (n_samples,)
            labels: array-like of shape (n_samples,), 1 (SAMA) or 0 (BEDA)
        """
        X = np.array(similarities).reshape(-1, 1)
        y = np.array(labels)
        
        self.calibrator.fit(X, y)
        self.is_fitted = True
        
    def predict_proba(self, similarities):
        """
        Mengubah cosine similarity menjadi probabilitas p_I.
        Args:
            similarities: array-like of shape (n_samples,)
        Returns:
            p_I: Probabilitas bahwa pasangan tersebut adalah identitas yang sama.
        """
        if not self.is_fitted:
            raise ValueError("Calibrator belum dilatih. Panggil fit() terlebih dahulu.")
            
        X = np.array(similarities).reshape(-1, 1)
        # predict_proba returns proba for class 0 and class 1. We want class 1 (SAME).
        probas = self.calibrator.predict_proba(X)
        return probas[:, 1]
