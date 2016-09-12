import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons

x, y = make_moons(n_samples=100, random_state=123)
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
x_kpca = kpca.fit_transform(x)

plt.scatter(x_kpca[y == 0, 0], x_kpca[y == 0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(x_kpca[y == 1, 0], x_kpca[y == 1, 1], color='blue', marker='o', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()