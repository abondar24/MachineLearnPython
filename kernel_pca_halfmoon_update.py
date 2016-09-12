import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn.datasets import make_moons


def rbf_kernel_pca(x, gamma, n_comps):
    """Radial-Basis Kernel PCA

    gamma - tuning param

    n_comps  - numbrer of principal components to return
    """

    # Calc pairwise squared Euclidian dists in mxn dimm dataset
    sq_dists = pdist(x, 'sqeuclidean')

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

    # Collect top k eigenvectors(proejcted samples)
    alphas = np.column_stack((eigvecs[:, -i] for i in range(1, n_comps+1)))

    # Collect corresp eigenvalues
    lambdas = [eigvals[-i] for i in range(1, n_comps+1)]

    return alphas, lambdas


def project_x(x_new, x, gamma, alphas, lamdas):
    pair_dist = np.array([np.sum((x_new - row)**2) for row in x])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas/ lambdas)

x, y = make_moons(n_samples=100, random_state=123)

alphas, lambdas = rbf_kernel_pca(x, gamma=15, n_comps=1)

x_new = x[25]
print('x_new: ', x_new)

x_proj = alphas[25]  # original projection
print('x_proj: ', x_proj)

x_reproj = project_x(x_new, x, gamma=15, alphas=alphas, lamdas=lambdas)
print('x_reproj: ', x_reproj)

plt.scatter(alphas[y == 0, 0], np.zeros(50), color='red', marker='^', alpha=0.5)
plt.scatter(alphas[y == 1, 0], np.zeros(50), color='blue', marker='o', alpha=0.5)
plt.scatter(x_proj, 0, color='black', label='original projection of point x(25)')
plt.scatter(x_proj, 0, color='green', marker='x', s=500)
plt.legend(scatterpoints=1)
plt.show()