# app.py
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.datasets import make_moons, make_circles, make_blobs, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, average_precision_score,
                             precision_recall_curve, roc_curve)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Playground: Classification", layout="wide")
st.title("Classification Playground")

# ---------- Reference links ----------
DATASET_LINKS = {
    "Iris":                    ("UCI / Wikipedia", "https://en.wikipedia.org/wiki/Iris_flower_data_set"),
    "Breast Cancer":           ("sklearn docs",    "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html"),
    "Wine":                    ("sklearn docs",    "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html"),
    "Digits (0\u20139)":           ("sklearn docs",    "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html"),
    "Synthetic: Moons":        ("sklearn docs",    "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html"),
    "Synthetic: Circles":      ("sklearn docs",    "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html"),
    "Synthetic: Blobs":        ("sklearn docs",    "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html"),
    "Synthetic: XOR":          ("XOR problem",     "https://en.wikipedia.org/wiki/Exclusive_or#Computation"),
    "Synthetic: Classification":("sklearn docs",   "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html"),
}
MODEL_LINKS = {
    "Logistic Regression": ("sklearn docs", "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"),
    "kNN":                 ("sklearn docs", "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html"),
    "Decision Tree":       ("sklearn docs", "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html"),
    "Random Forest":       ("sklearn docs", "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"),
    "SVM (RBF)":           ("sklearn docs", "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html"),
    "SVM (Linear)":        ("sklearn docs", "https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html"),
    "Na\u00efve Bayes":       ("sklearn docs", "https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html"),
    "Gradient Boosting":   ("sklearn docs", "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html"),
    "AdaBoost":            ("sklearn docs", "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html"),
    "MLP Neural Network":  ("sklearn docs", "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html"),
}

# ---------- Sidebar: data ----------
st.sidebar.header("1) Data")
data_choice = st.sidebar.selectbox("Dataset", [
    "Iris",
    "Breast Cancer",
    "Wine",
    "Digits (0–9)",
    "Synthetic: Moons",
    "Synthetic: Circles",
    "Synthetic: Blobs",
    "Synthetic: XOR",
    "Synthetic: Classification",
])

def load_data(name):
    if name == "Iris":
        d = datasets.load_iris()
        X, y, target_names = d.data, d.target, d.target_names
        feature_names = d.feature_names
    elif name == "Breast Cancer":
        d = datasets.load_breast_cancer()
        X, y, target_names = d.data, d.target, d.target_names
        feature_names = d.feature_names
    elif name == "Wine":
        d = datasets.load_wine()
        X, y, target_names = d.data, d.target, d.target_names
        feature_names = d.feature_names
    elif name == "Digits (0–9)":
        d = datasets.load_digits()
        X, y = d.data, d.target
        target_names = np.array([str(i) for i in range(10)])
        feature_names = np.array([f"pixel_{i}" for i in range(X.shape[1])])
    elif name == "Synthetic: Moons":
        X, y = make_moons(n_samples=600, noise=0.25, random_state=42)
        feature_names = ["x1", "x2"]; target_names = np.array(["class 0", "class 1"])
    elif name == "Synthetic: Circles":
        X, y = make_circles(n_samples=600, noise=0.1, factor=0.5, random_state=42)
        feature_names = ["x1", "x2"]; target_names = np.array(["class 0", "class 1"])
    elif name == "Synthetic: Blobs":
        X, y = make_blobs(n_samples=600, centers=4, cluster_std=1.2, random_state=42)
        feature_names = ["x1", "x2"]
        target_names = np.array([f"cluster {i}" for i in range(4)])
    elif name == "Synthetic: XOR":
        rng = np.random.RandomState(42)
        X = rng.randn(600, 2)
        y = (((X[:, 0] > 0) & (X[:, 1] > 0)) | ((X[:, 0] < 0) & (X[:, 1] < 0))).astype(int)
        feature_names = ["x1", "x2"]; target_names = np.array(["class 0", "class 1"])
    else:  # Synthetic: Classification
        X, y = make_classification(
            n_samples=600, n_features=10, n_informative=5, n_redundant=2,
            n_classes=3, n_clusters_per_class=1, random_state=42
        )
        feature_names = [f"feat_{i}" for i in range(10)]
        target_names = np.array(["class 0", "class 1", "class 2"])
    return np.array(X), np.array(y), np.array(feature_names), np.array(target_names)

X, y, feature_names, target_names = load_data(data_choice)
_ds_label, _ds_url = DATASET_LINKS[data_choice]
st.sidebar.caption(f"Learn more: [{_ds_label}]({_ds_url})")
st.write(f"Shape: {X.shape}, Classes: {np.unique(y)}")

# ---------- Sidebar: split & options ----------
st.sidebar.header("2) Split & Options")
st.sidebar.caption("[What is train/val/test splitting?](https://scikit-learn.org/stable/modules/cross_validation.html)")
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05,
    help="Fraction of the full dataset held out as the test set. The model never sees these samples during training.")
val_size = st.sidebar.slider("Validation size (from train)", 0.0, 0.3, 0.2, 0.05,
    help="Fraction of the training set used as a validation set for threshold tuning and curves. Set to 0 to skip.")
scale_needed = st.sidebar.checkbox("Standardize features (recommended for kNN/SVM/LogReg)", value=True,
    help="Applies sklearn StandardScaler (zero mean, unit variance). Essential for distance- and margin-based models. [Learn more](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling)")
class_weight = st.sidebar.checkbox("Use class_weight='balanced' (if supported)", value=False,
    help="Weights classes inversely proportional to their frequency, helping with imbalanced datasets. [Learn more](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)")

# Split train/val/test with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=42
)
if val_size > 0:
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=42
    )
else:
    X_val, y_val = X_test, y_test  # fallback for plotting threshold curves

# ---------- Sidebar: model ----------
st.sidebar.header("3) Model")
model_name = st.sidebar.selectbox(
    "Algorithm",
    [
        "Logistic Regression",
        "kNN",
        "Decision Tree",
        "Random Forest",
        "SVM (RBF)",
        "SVM (Linear)",
        "Naïve Bayes",
        "Gradient Boosting",
        "AdaBoost",
        "MLP Neural Network",
    ]
)
_m_label, _m_url = MODEL_LINKS[model_name]
st.sidebar.caption(f"[{_m_label} \u2197]({_m_url})")
def make_model(name):
    if name == "Logistic Regression":
        C = st.sidebar.slider("C (inverse regularization)", 0.01, 10.0, 1.0, 0.01,
            help="Smaller C = stronger L2 regularization (less overfitting). Larger C = less regularization (fits training data more closely).")
        solver = "lbfgs"
        kwargs = {"C": C, "max_iter": 1000}
        if class_weight: kwargs["class_weight"] = "balanced"
        base = LogisticRegression(**kwargs)
        needs_scaling = True
    elif name == "kNN":
        k = st.sidebar.slider("k (neighbors)", 1, 25, 5, 1,
            help="Number of nearest neighbours used for voting. Low k = complex, wiggly boundary (high variance). High k = smoother boundary (high bias).")
        base = KNeighborsClassifier(n_neighbors=k)
        needs_scaling = True
    elif name == "Decision Tree":
        depth = st.sidebar.slider("max_depth (0 = None)", 0, 20, 4, 1,
            help="Maximum depth of the tree. Deeper trees fit training data more closely but may overfit. 0 means unlimited depth.")
        min_leaf = st.sidebar.slider("min_samples_leaf", 1, 20, 1, 1,
            help="Minimum number of samples required at a leaf node. Higher values act as regularization and prevent overfitting.")
        kwargs = {"max_depth": (None if depth == 0 else depth), "min_samples_leaf": min_leaf, "random_state": 42}
        if class_weight: kwargs["class_weight"] = "balanced"
        base = DecisionTreeClassifier(**kwargs)
        needs_scaling = False
    elif name == "Random Forest":
        n = st.sidebar.slider("n_estimators", 10, 300, 100, 10,
            help="Number of decision trees in the forest. More trees reduce variance but increase training time. Performance usually plateaus after ~100-200 trees.")
        depth = st.sidebar.slider("max_depth (0 = None)", 0, 20, 0, 1,
            help="Maximum depth of each tree. 0 means unlimited. Shallower trees reduce overfitting at the cost of expressiveness.")
        kwargs = {"n_estimators": n, "max_depth": (None if depth == 0 else depth), "random_state": 42}
        if class_weight: kwargs["class_weight"] = "balanced"
        base = RandomForestClassifier(**kwargs)
        needs_scaling = False
    elif name == "SVM (RBF)":
        C = st.sidebar.slider("C", 0.01, 10.0, 1.0, 0.01,
            help="Regularization parameter. Smaller C = wider margin, more misclassifications tolerated (underfitting). Larger C = narrower margin, fewer misclassifications (overfitting).")
        gamma = st.sidebar.slider("gamma", 0.001, 2.0, 0.1, 0.001,
            help="RBF kernel width. Small gamma = wide kernel, smoother boundary (underfitting). Large gamma = narrow kernel, complex boundary (overfitting).")
        kwargs = {"C": C, "gamma": gamma, "probability": True, "random_state": 42}
        base = SVC(**kwargs)
        needs_scaling = True
    elif name == "SVM (Linear)":
        C = st.sidebar.slider("C (inverse regularization)", 0.01, 10.0, 1.0, 0.01,
            help="Smaller C = stronger regularization, wider margin. Larger C = less regularization, fits training data more closely.")
        max_iter = st.sidebar.slider("max_iter", 100, 5000, 1000, 100,
            help="Maximum number of iterations for the solver. Increase if you see convergence warnings.")
        kwargs = {"C": C, "max_iter": max_iter, "random_state": 42}
        if class_weight: kwargs["class_weight"] = "balanced"
        # Wrap LinearSVC in CalibratedClassifierCV for probability support
        base = CalibratedClassifierCV(LinearSVC(**kwargs))
        needs_scaling = True
    elif name == "Naïve Bayes":
        base = GaussianNB()
        needs_scaling = False
    elif name == "Gradient Boosting":
        n = st.sidebar.slider("n_estimators", 10, 300, 100, 10,
            help="Number of boosting stages (trees). More trees can improve accuracy but risk overfitting. Use with a small learning rate.")
        lr = st.sidebar.slider("learning_rate", 0.01, 1.0, 0.1, 0.01,
            help="Shrinks the contribution of each tree. Lower learning rate needs more trees but often generalises better (n_estimators and learning_rate trade off).")
        depth = st.sidebar.slider("max_depth", 1, 10, 3, 1,
            help="Maximum depth of each individual tree (weak learner). Shallower trees = simpler model; deeper trees = more complex interactions.")
        base = GradientBoostingClassifier(n_estimators=n, learning_rate=lr, max_depth=depth, random_state=42)
        needs_scaling = False
    elif name == "AdaBoost":
        n = st.sidebar.slider("n_estimators", 10, 300, 50, 10,
            help="Number of weak learners (decision stumps) to sequentially combine. More estimators can improve fit but may overfit noisy data.")
        lr = st.sidebar.slider("learning_rate", 0.01, 2.0, 1.0, 0.01,
            help="Shrinks the contribution of each weak learner. Lower values need more estimators. Trades off with n_estimators.")
        base = AdaBoostClassifier(n_estimators=n, learning_rate=lr, random_state=42)
        needs_scaling = False
    else:  # MLP Neural Network
        hidden = st.sidebar.selectbox("Hidden layer sizes", ["(64,)", "(128,)", "(64, 64)", "(128, 64)", "(256, 128, 64)"], index=2,
            help="Number and size of hidden layers. E.g. (64, 64) means two hidden layers of 64 neurons each. Larger/deeper networks have more capacity but need more data to generalise.")
        hidden = eval(hidden)
        alpha = st.sidebar.slider("alpha (L2 reg)", 0.0001, 1.0, 0.0001, 0.0001,
            help="L2 penalty on the weights. Higher alpha = stronger regularization, reduces overfitting.")
        lr_init = st.sidebar.slider("learning_rate_init", 0.0001, 0.1, 0.001, 0.0001,
            help="Initial step size for the Adam/SGD optimizer. Too high can cause divergence; too low can make training very slow.")
        base = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, learning_rate_init=lr_init,
                             max_iter=500, random_state=42)
        needs_scaling = True
    if scale_needed and needs_scaling:
        model = Pipeline([("scaler", StandardScaler()), ("clf", base)])
    else:
        model = base
    return model

model = make_model(model_name)

# ---------- Train ----------
model.fit(X_train, y_train)
proba_ok = hasattr(model, "predict_proba")
y_pred = model.predict(X_test)
st.subheader("Performance on Test Set")
st.caption(
    "[Precision, Recall & F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) · "
    "[Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) · "
    "[ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) · "
    "[Precision\u2013Recall AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)"
)
st.text(classification_report(y_test, y_pred, zero_division=0))

# ---------- Threshold tuning for probabilistic models ----------
if proba_ok:
    st.subheader("Threshold Tuning & Curves")
    st.caption("[What is a decision threshold?](https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/)")
    y_score = model.predict_proba(X_val)[:, 1] if len(np.unique(y)) == 2 else None
    if y_score is not None:
        thr = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01,
            help="Probability above which a sample is classified as the positive class. Default 0.5. Lower threshold = more positives predicted (higher recall, lower precision).")
        y_thr = (y_score >= thr).astype(int)
        st.write("Confusion matrix (Validation)")
        cm = confusion_matrix(y_val, y_thr)
        fig, ax = plt.subplots(1, 2, figsize=(12,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[0])
        ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("True"); ax[0].set_title("Confusion Matrix")

        # PR & ROC curves
        precision, recall, _ = precision_recall_curve(y_val, y_score)
        fpr, tpr, _ = roc_curve(y_val, y_score)
        pr_auc = average_precision_score(y_val, y_score)
        roc_auc = roc_auc_score(y_val, y_score)

        ax[1].plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
        ax[1].plot([0,1],[0,1],'--',color='gray',linewidth=1)
        ax[1].set_xlabel("False Positive Rate"); ax[1].set_ylabel("True Positive Rate"); ax[1].set_title("ROC Curve")
        ax[1].legend()
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(5,4))
        ax2.plot(recall, precision, color="purple", label=f"PR AUC={pr_auc:.3f}")
        ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title("Precision–Recall")
        ax2.legend()
        st.pyplot(fig2)

# ---------- Decision boundary with PCA (2D) ----------
st.subheader("Decision regions (PCA to 2D if needed)")
if X.shape[1] > 2:
    pca = PCA(n_components=2, random_state=42)
    X_vis = pca.fit_transform(X)
    # Retrain a copy on PCA space for visualization only
    vis_model = clone(model)
    vis_model.fit(X_vis, y)
else:
    X_vis = X.copy()
    vis_model = clone(model)
    vis_model.fit(X_vis, y)

# Mesh grid
x_min, x_max = X_vis[:,0].min()-0.5, X_vis[:,0].max()+0.5
y_min, y_max = X_vis[:,1].min()-0.5, X_vis[:,1].max()+0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
Z = vis_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig3, ax3 = plt.subplots(figsize=(6,5))
cs = ax3.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
scatter = ax3.scatter(X_vis[:,0], X_vis[:,1], c=y, cmap="coolwarm", edgecolor="k", s=25)
ax3.set_title("Decision Regions (2D view)")
st.pyplot(fig3)