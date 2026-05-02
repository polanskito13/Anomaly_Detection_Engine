# Anomaly Detection Engine
Multivariate anomaly detection system utilizing Mahalanobis Distance and Chi-squared distribution thresholds.

This engine implements a robust statistical framework to identify outliers in multivariate datasets. It is engineered for real-world applications where detecting unusual behavior is critical, such as:

- Banking & Financial Fraud
- Industrial Sensor Failures
- Network Intrusion Detection
- Biometric Data Validation
- Real-time System Monitoring

---

## Technical Highlights

### Mahalanobis Distance
Unlike standard Euclidean distance, the Mahalanobis distance accounts for:
- Correlation between variables
- Scaling differences between features
- The actual geometric shape of the data distribution

This allows the engine to detect outliers even when the data forms elongated ellipses rather than perfect circles.

### Chi-Squared Thresholding
The engine automatically calculates the critical threshold using the Chi-squared distribution:
- Degrees of freedom correspond to the number of input dimensions.
- Confidence levels define the detector's sensitivity (e.g., 0.95, 0.99).

### Scikit-learn Style API
The engine exposes a clean, professional interface:
- **fit(X)**: Learns the multivariate normal distribution.
- **predict(X)**: Determines if a data point is anomalous.
- **calculate_distance(X)**: Computes the raw Mahalanobis distance.
- **get_anomaly_score(X)**: Returns a normalized score (>1 indicates an anomaly).

---

## Project Structure

```text
6_Anomaly_Detection_Engine/
├── src/
│   └── detector.py
├── tests/
│   └── test_anomaly.py
├── docs/
├── main.py
├── README.md
├── requirements.txt
├── Dockerfile
├── LICENSE
└── .gitignore
```

---

## Testing & Installation

Unit tests validate that normal data points are not flagged and that extreme outliers are correctly identified.

**Run Tests:**
```bash
python -m pytest -s
```

**Run Demo:**
```bash
python main.py
```

## Requirements
- numpy
- scipy
- pytest

## License
MIT License — Free for academic and professional use.
