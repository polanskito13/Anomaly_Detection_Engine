import numpy as np
from src.detector import AnomalyEngine

def run_fraud_demo():
    print("\n" + "="*60)
    print("AI ENGINE #6: MULTIVARIATE ANOMALY DETECTION")
    print("Scenario: Banking Fraud Detection (Amount vs. Frequency)")
    print("="*60)
    
    np.random.seed(42)
    normal_data = np.random.multivariate_normal(
        mean=[50, 3600], 
        cov=[[100, 0.5], [0.5, 500]], 
        size=200
    )

    engine = AnomalyEngine(confidence_level=0.99)
    engine.fit(normal_data)

    test_cases = {
        "Normal Transaction": [52, 3580],
        "Fraud: Unusual Amount": [500, 3600],
        "Fraud: High Frequency": [50, 10],
        "Fraud: Critical Combined": [450, 15]
    }

    print(f"\n{'CASE':<30} | {'SCORE':<10} | {'STATUS'}")
    print("-" * 60)

    for name, data in test_cases.items():
        point = np.array(data)
        score = engine.get_anomaly_score(point)
        is_anomaly = engine.predict(point)
        
        status = "ANOMALY DETECTED" if is_anomaly else "OK"
        print(f"{name:<30} | {score:<10.4f} | {status}")

if __name__ == '__main__':
    run_fraud_demo()
