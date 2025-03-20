from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Training data
X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8]])

# Fit DBSCAN
db = DBSCAN(eps=1.5, min_samples=2).fit(X)

# New data point
new_point = np.array([[2, 2.5]])

# Extract core samples and their labels
core_samples = X[db.core_sample_indices_]
core_labels = db.labels_[db.core_sample_indices_]

# Find the nearest core point for the new point
nn = NearestNeighbors(n_neighbors=1).fit(core_samples)
distance, index = nn.kneighbors(new_point)

# Check if the new point is close enough (within eps)
if distance[0][0] <= db.eps:
    new_label = core_labels[index[0][0]]
    print(f"New point belongs to cluster: {new_label}")
else:
    print("New point is classified as noise.")
