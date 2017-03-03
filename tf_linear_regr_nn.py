import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle


def model(X, hidden_weights, hidden_bias, ow):
    hidden_layer = tf.nn.sigmoid(tf.matmul(X, hidden_weights) + b)
    return tf.matmul(hidden_layer, ow)


train_samples = 200
test_samples = 60

ds_X = np.linspace(-1, 1, train_samples + test_samples).transpose()
ds_Y = 0.4 * pow(ds_X, 2) + 2 * ds_X + np.random.randn(*ds_X.shape) * 0.22 + 0.8

plt.title('Original data')
plt.scatter(ds_X, ds_Y)
plt.show()

X = tf.placeholder("float")
Y = tf.placeholder("float")

# hidden layer
hw = tf.Variable(tf.random_normal([1, 10], stddev=0.1))

# output connection
ow = tf.Variable(tf.random_normal([10, 1], stddev=0.1))

# create bias
b = tf.Variable(tf.random_normal([10], stddev=0.1))

model_y = model(X, hw, b, ow)

# cost function
cost = tf.pow(model_y - Y, 2) / 2

# optimizer
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

# launch graph in session
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(1, 100):

        # randomize samples
        ds_X, ds_Y = shuffle(ds_X.transpose(), ds_Y)

        train_X, train_Y = ds_X[0:train_samples], ds_Y[0:train_samples]

        for x, y in zip(train_X, train_Y):
            sess.run(train_op, feed_dict={X: [[x]], Y: y})

        test_X, test_Y = ds_X[train_samples: train_samples + test_samples], \
                         ds_Y[0: train_samples:train_samples + test_samples]

        cost1 = 0.
        for x, y in zip(test_X, test_Y):
            cost1 += sess.run(cost, feed_dict={X: [[x]], Y: y}) / test_samples

        if i % 10 == 0:
            print("Average cost for epoch " + str(i) + ":" + str(cost1))
