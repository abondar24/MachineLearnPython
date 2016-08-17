from sklearn import tree

#
# input
# 0 - bumpy
# 1 - smoth
#

features = [[140, 1], [130, 1], [150, 0], [170, 0]]


# output - what belongs to each feature
# 0 - apple
# 1 - orange
labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()

# train a classifier
clf = clf.fit(features, labels)


print(clf.predict([[150, 0]]))