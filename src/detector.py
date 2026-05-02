import numpy as np
from scipy.stats import chi2

class AnomalyEngine:
    """
    Motor de Detección de Anomalías Multivariante.
    Utiliza la Distancia de Mahalanobis para identificar outliers considerando 
    la correlación entre múltiples variables.
    """
    
    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level
        self.mu = None        # Vector de medias
        self.inv_cov = None   # Inversa de la matriz de covarianza
        self.threshold = None # Umbral crítico basado en Chi-cuadrado

    def fit(self, X):
        """
        Entrena el modelo calculando la geometría de la distribución normal de los datos.
        """
        X = np.array(X)
        self.mu = np.mean(X, axis=0)
        # Calculamos la matriz de covarianza y su inversa (precisión)
        cov = np.cov(X, rowvar=False)
        self.inv_cov = np.linalg.inv(cov)
        
        # El umbral se define por la distribución Chi-cuadrado según los grados de libertad (columnas)
        df = X.shape[1] 
        self.threshold = chi2.ppf(self.confidence_level, df)
        print(f"[INFO] Motor entrenado. Umbral crítico (Chi2): {self.threshold:.4f}")

    def calculate_distance(self, X):
        """
        Calcula la Distancia de Mahalanobis para cada punto.
        """
        X = np.array(X)
        delta = X - self.mu
        # Fórmula: sqrt( (x-mu)^T * Inv_Cov * (x-mu) )
        m_dist = np.dot(np.dot(delta, self.inv_cov), delta.T)
        
        if m_dist.ndim > 1:
            return np.diagonal(m_dist)
        return m_dist

    def predict(self, X):
        """
        Determina si un punto es una anomalía.
        Retorna True para anomalías, False para datos normales.
        """
        distances = self.calculate_distance(X)
        return distances > self.threshold

    def get_anomaly_score(self, X):
        """
        Retorna un score de anomalía (distancia normalizada respecto al umbral).
        Valores > 1 son considerados anomalías.
        """
        distances = self.calculate_distance(X)
        return distances / self.threshold
