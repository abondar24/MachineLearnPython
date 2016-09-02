# implement a pipeline
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.cross_validation import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler


def plot_decision_regions(x, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot decision surface
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.xlim(xx2.min(), xx2.max())

    x_test, y_test = x[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0], y=x[y == cl, 1],
                    alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

    # highlight test samples
    x_test, y_test = x[test_idx, :], y[test_idx]
    plt.scatter(x_test[:, 0], x_test[:, 1], c='', alpha=1.0,
                linewidths=1, marker='o', s=55, label='test set')


iris = datasets.load_iris()

x = iris.data[:, [2, 3]]
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)
x_combined_std = np.vstack((x_train_std, x_test_std))
y_combined = np.hstack((y_train, y_test))

knn = KNeighborsClassifier()
knn.fit(x_train_std, y_train)

predictions = knn.predict(x_test)
print(predictions)
print(accuracy_score(y_test, predictions))

plot_decision_regions(x=x_combined_std, y=y_combined,
                      classifier=knn, test_idx=range(105, 150))
plt.xlabel('petal len[std]')
plt.ylabel('sepal len[std]')
plt.legend(loc='upper left')
plt.show()


knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(x_train_std, y_train)

predictions = knn.predict(x_test)
print(predictions)
print(accuracy_score(y_test, predictions))

plot_decision_regions(x=x_combined_std, y=y_combined,
                      classifier=knn, test_idx=range(105, 150))
plt.xlabel('petal len[std]')
plt.ylabel('sepal len[std]')
plt.legend(loc='upper left')
plt.show()