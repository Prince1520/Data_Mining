#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

def initialize_centroids(X, k):
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]
def initialize_centroids(X, k):
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

def kmeans(X, k, max_iters=100, tolerance=1e-4):
    centroids = initialize_centroids(X, k)
    prev_centroids = np.zeros(centroids.shape)
    clusters = np.zeros(X.shape[0])

    for _ in range(max_iters):
        for i in range(X.shape[0]):
            distances = np.linalg.norm(X[i] - centroids, axis=1)
            clusters[i] = np.argmin(distances)

        prev_centroids = centroids.copy()
        for i in range(k):
            if np.any(clusters == i): 
                centroids[i] = X[clusters == i].mean(axis=0)
        if np.linalg.norm(centroids - prev_centroids) < tolerance:
            break

    return centroids, clusters
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
k_values = range(1, 10)
inertia = []
silhouette_scores = []

for k in k_values:
    centroids, clusters = kmeans(X, k)
    inertia.append(np.sum((X - centroids[clusters.astype(int)])**2))
    if k > 1:
        silhouette_scores.append(silhouette_score(X, clusters))
    else:
        silhouette_scores.append(None)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_values, inertia, marker='o')
plt.title('Inertia vs Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')

plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score vs Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()
optimal_k = np.argmax(silhouette_scores[1:]) + 2 
final_centroids, final_clusters = kmeans(X, optimal_k)
plt.scatter(X[:, 0], X[:, 1], c=final_clusters, cmap='viridis', marker='o')
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title(f'Final Clustering with k={optimal_k}')
plt.legend()
plt.show()


# In[ ]:




