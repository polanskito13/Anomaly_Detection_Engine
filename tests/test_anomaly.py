import unittest
import numpy as np
from src.detector import AnomalyEngine

class TestAnomalyEngine(unittest.TestCase):
    def setUp(self):
        """Prepara un entorno de datos controlados para las pruebas."""
        # Generamos una nube de puntos normal (datos "legítimos")
        np.random.seed(42)
        self.normal_data = np.random.multivariate_normal(
            mean=[10, 10], 
            cov=[[1, 0.5], [0.5, 1]], 
            size=100
        )
        self.engine = AnomalyEngine(confidence_level=0.99)
        self.engine.fit(self.normal_data)

    def test_normal_point(self):
        """Prueba que un punto dentro de la distribución no sea marcado como anomalía."""
        point = np.array([10.5, 10.5])
        is_anomaly = self.engine.predict(point)
        self.assertFalse(is_anomaly, "Un punto cercano a la media NO debería ser anomalía.")

    def test_extreme_outlier(self):
        """Prueba que un punto muy alejado sea detectado correctamente."""
        # Un punto muy fuera de la "elipse" de confianza
        outlier = np.array([20.0, 20.0]) 
        is_anomaly = self.engine.predict(outlier)
        self.assertTrue(is_anomaly, "Un punto extremo DEBERÍA ser detectado como anomalía.")

    def test_anomaly_score(self):
        """Verifica que el score de anomalía sea mayor a 1 para outliers."""
        outlier = np.array([25.0, 5.0])
        score = self.engine.get_anomaly_score(outlier)
        self.assertGreater(score, 1, "El score de una anomalía debe ser mayor a 1.")

if __name__ == "__main__":
    unittest.main()
