import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets,cross_validation, metrics
from sklearn import preprocessing
import tensorflow.contrib.learn as learn

from keras.models import Sequential
from keras.layers import Dense

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
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)

# scale the data for convergency optimize
scaler = preprocessing.StandardScaler()

X_train = scaler.fit_transform(X_train)

# 2-layer fully connected DNN with 10 and 5 unit
model = Sequential()
model.add(Dense(10,input_dim=7,init='normal',activation='relu'))
model.add(Dense(5,init='normal',activation='relu'))
model.add(Dense(1,init='normal'))

model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(X_train,y_train,nb_epoch=1000,validation_split=0.33,shuffle=True,verbose=2)
