import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from matplotlib import cm
from sklearn.metrics import silhouette_samples

# data set of 150 random points
x, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)

plt.scatter(x[:, 0], x[:, 1], c='white', marker='o', s=50)
plt.grid()
plt.show()

km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
y_km = km.fit_predict(x)

plt.scatter(x[y_km == 0, 0], x[y_km == 0, 1], c='lightgreen', marker='s', s=50, label='cluster 1')
plt.scatter(x[y_km == 1, 0], x[y_km == 1, 1], c='orange', marker='o', s=50, label='cluster 2')
plt.scatter(x[y_km == 2, 0], x[y_km == 2, 1], c='lightblue', marker='v', s=50, label='cluster 1')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='red', marker='*', s=250, label='centroids ')

plt.legend()
plt.grid()
plt.show()

print('Distortion: %.2f' % km.inertia_)

# elbow method to reduce distortion
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, n_init=10, max_iter=300, random_state=0)
    km.fit(x)
    distortions.append(km.inertia_)

plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of cluster')
plt.ylabel('Distortion')
plt.show()

# quantifying qual by silhouette plots
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(x, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []

for i,c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[ y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(i / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color='red', linestyle='--')
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show()