import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from matplotlib.ticker import FormatStrFormatter


def rbf_kernel_pca(x, gamma, n_comps):
    """Radial-Basis Kernel PCA

    gamma - tuning param

    n_comps  - numbrer of principal components to return
    """

    # Calc pairwise squared Euclidian dists in mxn dimm dataset
    sq_dists = pdist(x,'sqeuclidean')

    # Convert pairwise squared Eculidian dist into a square matrix
    mat_sq_dists = squareform(sq_dists)

    # Compute the symetric kernel matrix
    k = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix
    n = k.shape[0]
    one_n = np.ones((n, n)) / n
    k = k - one_n.dot(k) - k.dot(one_n) + one_n.dot(k).dot(one_n)

    # Obtain eigenpairs from the centered kernel matrix
    eigvals, eigvecs = eigh(k)

    # projected ds
    x_pc = np.column_stack((eigvecs[:, -i]
                            for i in range(1, n_comps + 1)))

    return x_pc


x, y = make_moons(n_samples=100, random_state=123)
plt.scatter(x[y == 0, 0], x[y == 0, 1],
            color='red', marker='^', alpha=0.5)
plt.scatter(x[y == 1, 0], x[y == 1, 1],
            color='blue', marker='o', alpha=0.5)
plt.show()

scikit_pca = PCA(n_components=2)
x_spca = scikit_pca.fit_transform(x)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))

ax[0].scatter(x_spca[y == 0, 0], x_spca[y == 0, 1],
              color='red', marker='^', alpha=0.5)
ax[0].scatter(x_spca[y == 1, 0], x_spca[y == 1, 1],
              color='blue', marker='o', alpha=0.5)
ax[1].scatter(x_spca[y == 0, 0], np.zeros((50, 1)) + 0.02,
              color='red', marker='^', alpha=0.5)
ax[1].scatter(x_spca[y == 1, 0], np.zeros((50, 1)) - 0.02,
              color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()

x_kpca = rbf_kernel_pca(x, gamma=15, n_comps=2)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
ax[0].scatter(x_kpca[y == 0, 0], x_kpca[y == 0, 1],
              color='red', marker='^', alpha=0.5)
ax[0].scatter(x_kpca[y == 1, 0], x_kpca[y == 1, 1],
              color='blue', marker='o', alpha=0.5)
ax[1].scatter(x_kpca[y == 0, 0], np.zeros((50, 1)) + 0.02,
              color='red', marker='^', alpha=0.5)
ax[1].scatter(x_kpca[y == 1, 0], np.zeros((50, 1)) - 0.02,
              color='blue', marker='o', alpha=0.5)


ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
plt.show()