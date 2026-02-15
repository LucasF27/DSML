# Interactive DSML Visualizations

This repository contains interactive Python scripts to visualize and understand common Data Science and Machine Learning algorithms. These tools are built using `matplotlib` and `scikit-learn`.

## Files

### 1. `dbscan_interactive.py`
An interactive visualization of the **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) algorithm.
- **Features**:
    - Adjust parameters like `epsilon` and `min_samples`.
    - Switch between different datasets: Moons, Blobs, and Circles.
    - Visualize how density-based clustering handles non-linear data shapes compared to centroid-based methods.

### 2. `kmeans_animation.py`
A visualization of the **K-Means Clustering** algorithm.
- **Features**:
    - Step-by-step animation or interactive control of the clustering process.
    - Visualize centroid initialization, assignment steps, and centroid updates.
    - See how the algorithm converges on blob-based datasets.

### 3. `pca_interactive.py`
An interactive tool for **Principal Component Analysis (PCA)**.
- **Features**:
    - Explore dimensionality reduction on different datasets (High-dimensional blobs, Correlated 2D, Overlapping 3D).
    - Visualize how PCA projects data onto principal components to maximize variance.

## Requirements

To run these scripts, you need the following Python packages:
- `numpy`
- `matplotlib`
- `scikit-learn`

You can install them using:
```bash
pip install -r requirements.txt
```

## Usage

Run any of the scripts directly with Python:

```bash
python dbscan_interactive.py
# or
python kmeans_animation.py
# or
python pca_interactive.py
```
