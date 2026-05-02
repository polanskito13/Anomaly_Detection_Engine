# Anomaly Detection Engine  
Detección de anomalías multivariante utilizando Distancia de Mahalanobis y umbrales basados en Chi-cuadrado.

Este motor implementa un sistema estadístico robusto para identificar puntos atípicos en datos multivariantes. Está diseñado para aplicaciones reales donde es necesario detectar comportamientos inusuales, tales como:

- Transacciones bancarias
- Sensores industriales
- Tráfico de red
- Datos biométricos
- Monitoreo de sistemas en tiempo real

---

## Características Principales

### Distancia de Mahalanobis
A diferencia de la distancia euclidiana, la distancia de Mahalanobis considera:
- Correlación entre variables
- Diferencias de escala
- Forma real de la distribución

Esto permite detectar outliers incluso cuando los datos forman elipses en lugar de círculos.

### Umbral Basado en Chi-Cuadrado
El motor calcula automáticamente el umbral crítico utilizando la distribución Chi-cuadrado:
- Los grados de libertad corresponden al número de dimensiones
- El nivel de confianza define la severidad del detector (ej. 0.95, 0.99)

### API Estilo scikit-learn
El motor expone métodos simples y profesionales:
- **fit(X)**: aprende la distribución normal multivariante.
- **predict(X)**: determina si un punto es anómalo.
- **calculate_distance(X)**: calcula la distancia de Mahalanobis.
- **get_anomaly_score(X)**: devuelve un score normalizado (>1 indica anomalía).

---

## Estructura del Proyecto

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

## Pruebas Unitarias e Instalación

Las pruebas validan que puntos normales no sean marcados como anomalías y que outliers extremos sean detectados correctamente.

**Ejecutar pruebas:**
```bash
python -m pytest -s
```

**Ejecución de la Demo:**
```bash
python main.py
```

## Requerimientos
- numpy
- scipy
- pytest

## Licencia
MIT License — libre para uso académico y profesional.
