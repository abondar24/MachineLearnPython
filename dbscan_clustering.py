from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

x, y = make_moons(n_samples=500, noise=0.05, random_state=0)
plt.scatter(x[:, 0], x[:, 1])
plt.show()


# compare agg clusterning and k-means

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3))
km = KMeans(n_clusters=2, random_state=0)
y_km = km.fit_predict(x)
ax1.scatter(x[y_km == 0, 0], x[y_km == 0, 1], c='lightblue', marker='o', s=40, label='cluster 1')
ax1.scatter(x[y_km == 1, 0], x[y_km == 1, 1], c='red', marker='s', s=40, label='cluster 2')
ax1.set_title('K-means clustering')

ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
y_ac = ac.fit_predict(x)
ax2.scatter(x[y_ac == 0, 0], x[y_ac == 0, 1], c='lightblue', marker='o', s=40, label='cluster 1')
ax2.scatter(x[y_ac == 1, 0], x[y_ac == 1, 1], c='red', marker='s', s=40, label='cluster 2')
ax2.set_title('Agglomeratuve clustering')
plt.legend()
plt.show()

# DBSCAN
db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
y_db = db.fit_predict(x)
plt.scatter(x[y_db == 0, 0], x[y_db == 0, 1], c='lightblue', marker='o', s=40, label='cluster 1')
plt.scatter(x[y_db == 1, 0], x[y_db == 1, 1], c='red', marker='s', s=40, label='cluster 2')
plt.legend()
plt.show()
