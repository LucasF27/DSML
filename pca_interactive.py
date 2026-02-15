import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs, make_classification

# Initial State
current_data_mode = 0
data_modes = ['High Dim Blobs', 'Correlated 2D', 'Overlapping 3D']
X = None
y = None
pca_result = None

def generate_data():
    global X, y
    np.random.seed(np.random.randint(0, 1000))
    mode = data_modes[current_data_mode]
    
    if mode == 'High Dim Blobs':
        # 10 Features, plot will only show feat 1 vs 2 initially, which might look cluttered
        X, y = make_blobs(n_samples=300, n_features=10, centers=3, cluster_std=3.0, random_state=None)
        
    elif mode == 'Correlated 2D':
        # 2 Features, highly correlated, PCA should rotate to axis aligned
        X, y = make_blobs(n_samples=300, n_features=2, centers=2, cluster_std=1.0)
        # Stretch and rotate
        transform = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X, transform)
        
    elif mode == 'Overlapping 3D':
        # 3 Features, hard to see in 2D projection of just feat 1&2
        X, y = make_blobs(n_samples=400, n_features=3, centers=4, cluster_std=2.0)
        
    return X, y

# Generate initial
X, y = generate_data()

# Setup Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.2)

def update_plot():
    global X, y
    ax1.clear()
    ax2.clear()
    
    # Run PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    c_map = [colors[label % len(colors)] for label in y]
    
    # Plot 1: Original Features (Feature 0 vs Feature 1)
    ax1.set_title(f'Original Data (Feat 1 vs Feat 2)\nMode: {data_modes[current_data_mode]}')
    ax1.scatter(X[:, 0], X[:, 1], c=c_map, alpha=0.6, edgecolors='w')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.grid(alpha=0.3)
    
    # Plot 2: PCA Results (PC1 vs PC2)
    explained_var = pca.explained_variance_ratio_
    ax2.set_title(f'PCA Result (PC1 vs PC2)\nVar: {explained_var[0]:.2f}, {explained_var[1]:.2f}')
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=c_map, alpha=0.6, edgecolors='w')
    ax2.set_xlabel('Principal Component 1')
    ax2.set_ylabel('Principal Component 2')
    ax2.grid(alpha=0.3)
    
    plt.draw()

def next_dataset(event):
    global current_data_mode, X, y
    current_data_mode = (current_data_mode + 1) % len(data_modes)
    X, y = generate_data()
    update_plot()

def diff_sample(event):
    global X, y
    X, y = generate_data() # Keep mode, regen data
    update_plot()

# Controls
ax_next = plt.axes([0.35, 0.05, 0.15, 0.075])
ax_regen = plt.axes([0.55, 0.05, 0.15, 0.075])

btn_next = Button(ax_next, 'Next Data Type')
btn_regen = Button(ax_regen, 'New Sample')

btn_next.on_clicked(next_dataset)
btn_regen.on_clicked(diff_sample)

update_plot()
plt.show()
