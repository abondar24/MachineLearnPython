import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
              'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']


x = df[['LSTAT']].values
y = df['MEDV'].values

regr = LinearRegression()

# create polynomial features
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)

x_quad = quadratic.fit_transform(x)
x_cubic = cubic.fit_transform(x)

# linear fit
x_fit = np.arange(x.min(), x.max(), 1)[:, np.newaxis]
regr = regr.fit(x, y)
y_lin_fit = regr.predict(x_fit)
linear_r2 = r2_score(y, regr.predict(x))

# quadratic fit
regr = regr.fit(x_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(x_fit))
quadratic_r2 = r2_score(y, regr.predict(x_quad))

# cubic fit
regr = regr.fit(x_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(x_fit))
cubic_r2 = r2_score(y, regr.predict(x_cubic))

# plot res
plt.scatter(x, y, label='training points', color='lightgray')
plt.plot(x_fit, y_lin_fit, label='linear (d=1), $R^2=%.2f' % linear_r2, color='blue', lw=2, linestyle=':')
plt.plot(x_fit, y_quad_fit, label='quadratic (d=2), $R^2=%.2f' % quadratic_r2, color='red', lw=2, linestyle='-')
plt.plot(x_fit, y_cubic_fit, label='cubic (d=3), $R^2=%.2f' % cubic_r2, color='green', lw=2, linestyle='--')
plt.xlabel('% lower status of population [LSTAT]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper right')
plt.show()