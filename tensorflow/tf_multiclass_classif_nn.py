import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
from sklearn import preprocessing

sess = tf.InteractiveSession()

flags = tf.app.flags
FLAGS = flags.FLAGS

df = pd.read_csv('../data/wine.csv', header=0)
print(df.describe)

for i in range(1, 8):
    number = 420 + i
    ax1 = plt.subplot(number)
    ax1.locator_params(nbins=3)
    plt.title(list(df)[i])
    ax1.scatter(df[df.columns[i]], df['Wine'])

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

X = df[df.columns[1:13]].values
y = df['Wine'].values - 1
Y = tf.one_hot(indices=y, depth=3, on_value=1., off_value=0., axis=1, name="a").eval()
X, Y = shuffle(X, Y)

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

# create model
x = tf.placeholder(tf.float32, [None, 12])
W = tf.Variable(tf.zeros([12, 3]))
b = tf.Variable(tf.zeros([3]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 3])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# train
tf.global_variables_initializer().run()
for i in range(100):
    X, Y = shuffle(X, Y, random_state=1)

    X_tr = X[0:140, :]
    Y_tr = Y[0:140, :]

    X_t = X[140:178, :]
    Y_t = Y[140:178, :]

    X_tr, Y_tr = shuffle(X_tr, Y_tr, random_state=0)

    batch_xs, batch_ys = X_tr, Y_tr
    train_step.run({x: batch_xs, y_: batch_ys})
    cost = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})

    # test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval({x: X_t, y_: Y_t}))
