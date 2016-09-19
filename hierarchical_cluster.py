import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
x = np.random.random_sample([5, 3]) * 10
df = pd.DataFrame(x, columns=variables, index=labels)
print(df)

# euc_dist between each pair of samples - not good
row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')), columns=labels, index=labels)
print(row_dist)

# euc_dist between each pair of samples - good
row_clusters = linkage(df.values, metric='euclidean', method='complete')

print(pd.DataFrame(row_clusters,
             columns=['row label 1',
                      'row label 2',
                      'distance',
                      'no. of items in cluster'],
             index=['cluster %d' % (i+1) for i in range(row_clusters.shape[0])]))

row_dendr = dendrogram(row_clusters, labels=labels)
plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()

# dendogram + heat_map
# create a new figure object and define x,y pos + rotate dendogram 90-deg clockwise
fig = plt.figure(figsize=(8, 8))
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, orientation='left')

# reorder df according to cluster labels
df_rowclust = df.ix[row_dendr['leaves'][::-1]]

# construct heatmap
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')

# modify aesthetics of heatmap
axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)

axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()

# agg clustering scikit
ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
labels = ac.fit_predict(x)
print('Cluster labels: %s' % labels)


