import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error


TIMESTAMPS = 5
RNN_LAYERS = [{"steps": TIMESTAMPS}]
DENSE_LAYERS = None
BATCH_SIZE = 100
TRAINING_STEPS = 10000
PRINT_STEPS = TRAINING_STEPS / 100


def lstm_model(time_steps, rnn_layers, dense_layers=None):
    def lstm_cells(layers):
        return [tf.contrib.rnn.BasicLSTMCell(layer['steps'], state_is_tuple=True) for layer in layers]

    def dnn_layers(input_layers, layers):
        return input_layers

    def _lstm_model(X, y):
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells(rnn_layers), state_is_tuple=True)
        x_ = tf.unstack(X, axis=1)
        output, layers = tf.contrib.rnn.static_rnn(stacked_lstm, x_, dtype=dtypes.float64)
        output = dnn_layers(output[-1], dense_layers)
        return learn.models.linear_regression(output, y)

    return _lstm_model


regressor = learn.Estimator(model_fn=lstm_model(TIMESTAMPS, RNN_LAYERS, DENSE_LAYERS))

df = pd.read_csv("elec_load.csv", error_bad_lines=False)
plt.subplot()
plot_test, = plt.plot(df.values[:1500], label='Load')
plt.legend(handles=[plot_test])
plt.show()

print(df.describe())
array = (df.values - 147.0) / 339.0
plt.subplot()
plot_test, = plt.plot(array[:1500], label='Normalized Load')
plt.legend(handles=[plot_test])
plt.show()

list_X = []
list_y = []
X = {}
y = {}

for i in range(0, len(array) - 6):
    list_X.append(array[i:i + 5].reshape([5, 1]))
    list_y.append(array[i + 6])

array_X = np.array(list_X)
array_y = np.array(list_y)

X['train'] = array_X[0:12000]
X['test'] = array_X[12000:13000]
X['val'] = array_X[13000:14000]

y['train'] = array_y[0:12000]
y['test'] = array_y[12000:13000]
y['val'] = array_y[13000:14000]

# create a lstm instance and validation monitor
val_mon = learn.monitors.ValidationMonitor(X['val'], y['val'], every_n_steps=PRINT_STEPS, early_stopping_rounds=1000)


regressor.fit(X['train'], y['train'], steps=TRAINING_STEPS, batch_size=BATCH_SIZE, monitors=[val_mon])

predicted = regressor.predict(X['test'])
rmse = np.sqrt(((predicted - y['test']) ** 2).mean(axis=0))
score = mean_squared_error(predicted, y['test'])
print("MSE: %f" % score)

plt.subplot()
plot_predicted = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(y['test'], label='test')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()
