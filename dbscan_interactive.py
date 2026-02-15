import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_blobs, make_circles

# Initial Parameters
eps = 0.3
min_samples = 5
dataset_type = 'moons' # 'moons', 'blobs', 'circles'
n_samples = 300
seed = 42

# Colors for clusters
colors = ['blue', 'green', 'purple', 'orange', 'olive', 'red', 'cyan', 'magenta', 'yellow', 'brown']

# Data Generation
def generate_data():
    np.random.seed(seed)
    if dataset_type == 'moons':
        X, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=seed)
    elif dataset_type == 'circles':
        X, _ = make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)
    else: # blobs
        X, _ = make_blobs(n_samples=n_samples, centers=4, cluster_std=0.60, random_state=seed)
    return X

X = generate_data()

# Initialize Figure
fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(bottom=0.35)

def update_plot():
    ax.clear()
    ax.set_title(f'DBSCAN Result (eps={eps}, min_samples={min_samples})', fontsize=16)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    # Run DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    
    # Identify Core, Border, and Noise points
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    
    unique_labels = set(labels)
    
    # Plotting
    for k in unique_labels:
        col = 'black'
        if k != -1:
            col = colors[k % len(colors)]
        
        class_member_mask = (labels == k)
        
        # Plot Core Points (Large colored circles)
        if k != -1:
            xy = X[class_member_mask & core_samples_mask]
            ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14, label=f'Cluster {k} Core' if k==0 else "")
        
        # Plot Border Points (Small colored circles)
        if k != -1:
            xy = X[class_member_mask & ~core_samples_mask]
            ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6, label=f'Cluster {k} Border' if k==0 else "")
            
        # Plot Noise Points (Small black circles)
        if k == -1:
            xy = X[class_member_mask]
            ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor='black', markeredgecolor='k', markersize=6, label='Noise')
            
    # Legend handling: deduplicate labels
    handles, labels_plot = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_plot, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    ax.grid(alpha=0.3)
    plt.draw()

# Callbacks
def submit_eps(text):
    global eps
    try:
        val = float(text)
        if val > 0:
            eps = val
            update_plot()
    except ValueError:
        pass

def submit_min_samples(text):
    global min_samples
    try:
        val = int(text)
        if val > 0:
            min_samples = val
            update_plot()
    except ValueError:
        pass

def toggle_dataset(event):
    global dataset_type, X
    if dataset_type == 'moons':
        dataset_type = 'blobs'
    elif dataset_type == 'blobs':
        dataset_type = 'circles'
    else:
        dataset_type = 'moons'
    X = generate_data()
    update_plot()

def recalculate(event):
    global eps, min_samples
    try:
        val_eps = float(text_eps.text)
        if val_eps > 0:
            eps = val_eps
    except ValueError:
        pass
        
    try:
        val_min = int(text_min_spl.text)
        if val_min > 0:
            min_samples = val_min
    except ValueError:
        pass
    
    update_plot()

# Widgets
ax_eps = plt.axes([0.25, 0.2, 0.15, 0.05])
ax_min_spl = plt.axes([0.65, 0.2, 0.15, 0.05])
ax_dataset = plt.axes([0.25, 0.1, 0.2, 0.075])
ax_recalc = plt.axes([0.55, 0.1, 0.2, 0.075])

text_eps = TextBox(ax_eps, 'Epsilon:', initial=str(eps))
text_min_spl = TextBox(ax_min_spl, 'Min Samples:', initial=str(min_samples))
btn_dataset = Button(ax_dataset, 'Change Dataset')
btn_recalc = Button(ax_recalc, 'Recalculate')

text_eps.on_submit(submit_eps)
text_min_spl.on_submit(submit_min_samples)
btn_dataset.on_clicked(toggle_dataset)
btn_recalc.on_clicked(recalculate)

update_plot()
plt.show()
