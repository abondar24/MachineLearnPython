import numpy as np
import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO


iris = load_iris()

# indexes to remove from dataset
test_idx = [0, 50, 100]


# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data - exclude from dataset
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))


print(test_data[0], test_target[0])
print(iris.feature_names, iris.target_names)

print(test_data[1], test_target[1])
print(iris.feature_names, iris.target_names)


# tree visualization
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True,rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")