import numpy as np
import matplotlib.pyplot as plt
import pandas_demos as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
wine.columns = ['Class label', 'Alcohol',
                'Malic acid', 'Ash',
                'Alcanity of ash', 'Magnesium',
                'Total phenols', 'Flavanoids',
                'Nonflavanoid phenols',
                'Froanthocyaninca',
                'Color intencity', 'Hue',
                'OD280/OD135 of diluted water',
                'Proline']



x, y = wine.iloc[:, 1:].values, wine.iloc[:, 0].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

stdsc = StandardScaler()
x_train_std = stdsc.fit_transform(x_train)
x_test_std = stdsc.transform(x_test)

feat_labels = wine.columns[1:]
forest = RandomForestClassifier(n_estimators=10000,
                                random_state=0,
                                n_jobs=-1)

forest.fit(x_train, y_train)
importances = forest.feature_importances_
incides = np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f], importances[incides[f]]))


plt.title('Feature Importances')
plt.bar(range(x_train.shape[1]),
        importances[incides],
        color='lightblue',
        align='center')
plt.xticks(range(x_train.shape[1]),
           feat_labels, rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.tight_layout()
plt.show()