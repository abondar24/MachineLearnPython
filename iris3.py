# implement own classifier
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

def euc_distance(a,b):
    return distance.euclidean(a, b)

# KNN - key nearest neighbours
class ScrappyKNN():
    def fit(self, x_tr, y_tr):
        self.x_tr = x_tr
        self.y_tr = y_tr

    def predict(self, x_tst):
        res = []
        for row in x_tst:
            label = self.closest(row)
            res.append(label)
        return res

    def closest(self,row):
        best_dist = euc_distance(row, self.x_tr[0])
        best_index = 0
        for i in range(1, len(self.x_tr)):
            dist = euc_distance(row,self.x_tr[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_tr[best_index]



iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

cls = ScrappyKNN()
cls.fit(x_train, y_train)
predictions = cls.predict(x_test)
print(predictions)
print(accuracy_score(y_test, predictions))
