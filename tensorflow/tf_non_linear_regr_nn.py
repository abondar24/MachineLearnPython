import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from sklearn import model_selection, metrics
from sklearn import preprocessing
import tensorflow.contrib.learn as learn

df = pd.read_csv('../data/mpg.csv', header=0)

# convert the displacement column as float
df['displacement'] = df['displacement'].astype(float)

# we get data columns from the dataset
# ignore first and last for x
X = df[df.columns[1:8]]
y = df['mpg']

f, ax1 = plt.subplots()

for i in range(1, 8):
    number = 420 + i
    ax1.locator_params(nbins=3)
    ax1 = plt.subplot(number)
    plt.title(list(df)[i])
    ax1.scatter(df[df.columns[i]], y)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

# split datasets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)

# scale the data for convergency optimize
scaler = preprocessing.StandardScaler()

X_train = scaler.fit_transform(X_train)

# 2-layer fully connected DNN with 10 and 5 unit

feature_columns = learn.infer_real_valued_columns_from_input(X)

regressor = learn.DNNRegressor(hidden_units=[10, 5], feature_columns=feature_columns,
                               optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.051))

regressor.fit(X_train, y_train, batch_size=1, steps=500)

# get some metrics based on the X and Y test data
predictions = list(regressor.predict(scaler.transform(X_test), as_iterable=True))
score = metrics.mean_squared_error(predictions, y_test)

print(" Total Mean Squared Error:" + str(score))
