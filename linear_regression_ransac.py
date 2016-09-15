import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression, RANSACRegressor


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
              'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

x = df[['RM']].values
y = df['MEDV'].values

# ransac -  Radom Sample Consensus
ransac = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=50,
                         residual_metric=lambda x: np.sum(np.abs(x), axis=1),
                         residual_threshold=5.0, random_state=0)
ransac.fit(x, y)

inliner_mask = ransac.inlier_mask_
outliner_mask = np.logical_not(inliner_mask)
line_x = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_x[:, np.newaxis])

plt.scatter(x[inliner_mask], y[inliner_mask], c='blue', marker='o', label='inliners')
plt.scatter(x[outliner_mask], y[outliner_mask], c='lightgreen', marker='s', label='outliners')
plt.plot(line_x, line_y_ransac, color='red')
plt.xlabel('Average number of rooms[RM]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper left')
plt.show()

print('Slope: %.3f' % ransac.estimator_.coef_[0])
print('Intercept: %.3f' % ransac.estimator_.intercept_)

