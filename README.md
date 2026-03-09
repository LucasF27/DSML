# Interactive DSML Visualizations

Interactive Python tools for exploring Data Science and Machine Learning concepts. Built with `matplotlib`, `scikit-learn`, and `streamlit`.

## Files

### 1. `classification.py` — Classification Playground (Streamlit app)

A fully interactive browser-based app for experimenting with supervised classification.

**Datasets**
| Dataset | Type | Classes | Features |
|---|---|---|---|
| Iris | Real | 3 | 4 |
| Breast Cancer | Real | 2 | 30 |
| Wine | Real | 3 | 13 |
| Digits (0–9) | Real | 10 | 64 |
| Synthetic: Moons | Synthetic | 2 | 2 |
| Synthetic: Circles | Synthetic | 2 | 2 |
| Synthetic: Blobs | Synthetic | 4 | 2 |
| Synthetic: XOR | Synthetic | 2 | 2 |
| Synthetic: Classification | Synthetic | 3 | 10 |

**Models**
| Algorithm | Key hyperparameters |
|---|---|
| Logistic Regression | C |
| k-Nearest Neighbours | k |
| Decision Tree | max_depth, min_samples_leaf |
| Random Forest | n_estimators, max_depth |
| SVM (RBF kernel) | C, gamma |
| SVM (Linear kernel) | C, max_iter |
| Naïve Bayes | — |
| Gradient Boosting | n_estimators, learning_rate, max_depth |
| AdaBoost | n_estimators, learning_rate |
| MLP Neural Network | hidden layer sizes, alpha, learning_rate_init |

**Features**
- Stratified train / validation / test splitting with adjustable sizes
- Optional feature standardisation via `StandardScaler`
- Optional `class_weight='balanced'` for imbalanced data
- Classification report (precision, recall, F1) on the test set
- Decision threshold tuning with live confusion matrix (binary tasks)
- ROC curve + AUC and Precision–Recall curve + AUC (binary tasks)
- Decision region plot — uses PCA to 2D for high-dimensional datasets
- Contextual help tooltips and links to sklearn docs throughout

**Run:**
```bash
streamlit run classification.py
```

---

### 2. `kmeans_animation.py`

A step-by-step visualization of the **K-Means Clustering** algorithm.

- Animate centroid initialization, assignment, and update steps.
- Watch the algorithm converge on blob-based datasets.

```bash
python kmeans_animation.py
```

---

### 3. `dbscan_interactive.py`

An interactive visualization of **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise).

- Adjust `epsilon` and `min_samples` interactively.
- Switch between Moons, Blobs, and Circles datasets.
- Compare density-based vs centroid-based clustering on non-linear shapes.

```bash
python dbscan_interactive.py
```

---

### 4. `pca_interactive.py`

An interactive tool for **Principal Component Analysis (PCA)**.

- Explore dimensionality reduction on high-dimensional blobs, correlated 2D, and overlapping 3D datasets.
- Visualize how PCA projects data onto principal components to maximise variance.

```bash
python pca_interactive.py
```

---

### 5. `time_series_app.py` — Forecast Playground (Streamlit app)

An interactive browser-based app for teaching **time-series forecasting** on synthetic monthly temperature data (Nottingham, 2015–2024).

**Models**
| Model | Description |
|---|---|
| Mean | Global mean of the training set |
| Naïve | Last observed value repeated |
| Seasonal Naïve | Value from the same month one year prior |
| Moving Average | Rolling mean over a configurable window |
| Linear Regression | Lag-feature regression with sklearn |
| Random Forest | Ensemble of lag-feature regressors |

**Features**
- Configurable train / test split
- MAE and RMSE metrics displayed live
- Forecast vs actual chart with confidence shading
- Teaching notes on bias–variance and stationarity

**Run:**
```bash
streamlit run time_series_app.py
```

---

### 6. `anomaly_app.py` — Anomaly Detector Playground (Streamlit app)

An interactive browser-based app for teaching **anomaly detection** in time-series data using a synthetic heart-rate sensor trace (480 minutes, 12 injected spikes).

**Methods**
| Method | Description |
|---|---|
| Absolute threshold | Fixed upper / lower bounds; flags any crossing |
| Rolling z-score | Adaptive local mean ± k·σ band over a sliding window |

**Features**
- Configurable detection method, rolling window, and threshold
- Ground-truth toggle to reveal / hide injected anomaly locations
- Live confusion-matrix metrics: TP, FP, FN, Precision, Recall, F1
- Z-score panel below the main chart
- Teaching notes on the precision–recall trade-off and domain cost asymmetry

**Run:**
```bash
streamlit run anomaly_app.py
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `streamlit`, `seaborn`.
