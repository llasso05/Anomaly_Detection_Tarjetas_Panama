# Isolation Forest Anomaly Detection

This project uses the **Isolation Forest** algorithm to detect anomalies (outliers) in a given dataset. The model is implemented using `scikit-learn` and supports hyperparameter tuning via `GridSearchCV`.

---

##  Overview

Isolation Forest is an **unsupervised machine learning algorithm** specifically designed for **anomaly detection**. It works by isolating observations using random partitions and leverages the idea that anomalous points are more susceptible to isolation.

Anomalies are detected based on how few splits are needed to isolate a data point in randomly constructed binary trees. The fewer the splits, the more likely the point is anomalous.

---

##  Model Parameters

The following grid was used for hyperparameter tuning:

```python
param_grid = {
    'model__n_estimators': [50, 100],
    'model__max_samples': ['auto', 0.8],
    'model__contamination': [0.05, 0.1],
    'model__max_features': [1.0]
}
