import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from sklearn.datasets import make_blobs

# Initialize K-Means parameters
k = 5  # Number of clusters
n_blobs = 300  # Number of data points
n_blobs_centers = 5  # Number of blob centers (for data generation)

colors = ['blue', 'green', 'purple', 'orange', 'olive', 'red', 'cyan', 'magenta', 'yellow', 'brown']

# Function to assign points to the nearest centroid
def assign_clusters(data, centroids):
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

# Function to update centroids based on cluster assignments
def update_centroids(data, labels, k):
    return np.array([data[labels == i].mean(axis=0) for i in range(k)])

# Function to generate data and iterations
def generate_iterations(k, n_blobs_centers, seed=27, n_samples=300):
    np.random.seed(seed)
    data, _ = make_blobs(n_samples=n_samples, centers=n_blobs_centers, cluster_std=0.6)
    
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    
    # Prepare for step-by-step control
    iterations = []
    for _ in range(10):  # Maximum of 10 iterations
        labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, labels, k)
        iterations.append((centroids.copy(), labels.copy()))
        if np.all(centroids == new_centroids):  # Stop if centroids don't change
            break
        centroids = new_centroids
    
    return data, iterations

# Generate initial data
seed = 27  # Initial random seed for reproducibility
data, iterations = generate_iterations(k, n_blobs_centers, seed=seed)

# Initialize plot
fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(bottom=0.35)
current_step = [0]  # Use a mutable object to track the current step

# Store centroid history for transparency effect
centroid_history = [[] for _ in range(k)]

# Function to update the plot
def update_plot(step):
    ax.clear()
    ax.set_title('K-Means Clustering Step-by-Step', fontsize=16)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    
    # Get current iteration's centroids and labels
    centroids, labels = iterations[step]
    
    # Plot data points with their cluster colors
    for cluster_idx in range(k):
        cluster_points = data[labels == cluster_idx]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[cluster_idx], label=f'Cluster {cluster_idx + 1}', alpha=0.6)
        
    # Plot centroid movement history with decreasing transparency
    for cluster_idx in range(k):
        centroid_history[cluster_idx].append(centroids[cluster_idx])  # Add current centroid to history
    
    # Plot all previous centroid positions with decreasing alpha (transparency)
    for cluster_idx in range(k):
        for i, past_centroid in enumerate(centroid_history[cluster_idx]):
            alpha = max(0.1, 1 - (len(centroid_history[cluster_idx]) - i - 1) * 0.2)  # Older points are more transparent
            ax.scatter(past_centroid[0], past_centroid[1], c=colors[cluster_idx], marker='X', s=100, alpha=alpha)

    # Plot current centroids prominently
    ax.scatter(centroids[:, 0], centroids[:, 1], c=colors[:k], marker='X', s=200, edgecolor='black', label='Centroid')
    
    ax.legend()
    ax.grid(alpha=0.3)
    plt.draw()

# Button click handlers
def next_step(event):
    if current_step[0] < len(iterations) - 1:
        current_step[0] += 1
        update_plot(current_step[0])

def prev_step(event):
    if current_step[0] > 0:
        current_step[0] -= 1
        update_plot(current_step[0])

def reset(event):
    global data, iterations, k, n_blobs_centers, centroid_history, seed
    current_step[0] = 0
    centroid_history = [[] for _ in range(k)]
    data, iterations = generate_iterations(k, n_blobs_centers, seed=seed)
    update_plot(current_step[0])

def submit_clusters(text):
    global k, data, iterations, centroid_history, seed
    try:
        new_k = int(text)
        if new_k > 0 and new_k <= len(colors):
            k = new_k
            current_step[0] = 0
            centroid_history = [[] for _ in range(k)]
            data, iterations = generate_iterations(k, n_blobs_centers, seed=seed)
            update_plot(current_step[0])
        else:
            print(f"Number of clusters must be between 1 and {len(colors)}")
    except ValueError:
        print("Please enter a valid integer for clusters")

def submit_blobs(text):
    global data, iterations, centroid_history, n_blobs_centers, seed
    try:
        new_n_blobs_centers = int(text)
        if new_n_blobs_centers > 0:
            n_blobs_centers = new_n_blobs_centers
            current_step[0] = 0
            centroid_history = [[] for _ in range(k)]
            data, iterations = generate_iterations(k, n_blobs_centers, seed=seed)
            update_plot(current_step[0])
        else:
            print("Number of blob centers must be greater than 0")
    except ValueError:
        print("Please enter a valid integer for blob centers")

def submit_seed(text):
    global data, iterations, centroid_history, seed
    try:
        new_seed = int(text)
        current_step[0] = 0
        centroid_history = [[] for _ in range(k)]
        seed = new_seed
        data, iterations = generate_iterations(k, n_blobs_centers, seed=seed)
        update_plot(current_step[0])
    except ValueError:
        print("Please enter a valid integer for random seed")

# Add buttons and text boxes
axprev = plt.axes([0.2, 0.2, 0.1, 0.075])
axnext = plt.axes([0.31, 0.2, 0.1, 0.075])
axreset = plt.axes([0.7, 0.2, 0.1, 0.075])
axclusters = plt.axes([0.425, 0.13, 0.15, 0.05])
axblobs = plt.axes([0.2, 0.13, 0.15, 0.05])
axseed = plt.axes([0.7, 0.13, 0.15, 0.05])

btn_next = Button(axnext, 'Next')
btn_prev = Button(axprev, 'Previous')
btn_reset = Button(axreset, 'Reset')

text_clusters = TextBox(axclusters, 'Clusters:', initial=str(k))
text_blobs = TextBox(axblobs, 'Blob Centers:', initial=str(n_blobs_centers))
text_seed = TextBox(axseed, 'Random Seed:', initial='27')

btn_next.on_clicked(next_step)
btn_prev.on_clicked(prev_step)
btn_reset.on_clicked(reset)
text_clusters.on_submit(submit_clusters)
text_blobs.on_submit(submit_blobs)
text_seed.on_submit(submit_seed)

# Show the initial plot
update_plot(current_step[0])
plt.show()
