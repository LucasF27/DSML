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

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `streamlit`, `seaborn`.
