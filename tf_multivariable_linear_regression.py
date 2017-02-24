import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.learn as skflow
import pandas as pd

from  sklearn.utils import shuffle

df = pd.read_csv('boston.csv', header=0)
print(df.describe())

f, ax1 = plt.subplots()
plt.figure()
y = df['MEDV']

for i in range(1, 8):
    number = 420 + i
    ax1.locator_params(nbins=3)
    ax1 = plt.subplot(number)
    plt.title(list(df)[i])
    # print a scatter draw of datapoints
    ax1.scatter(df[df.columns[i]], y)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

X = tf.placeholder("float", name="X")
Y = tf.placeholder("float", name="Y")

with tf.name_scope("Model"):
    w = tf.Variable(tf.random_normal([2], stddev=0.01), name="b0")
    b = tf.Variable(tf.random_normal([2], stddev=0.01), name="b1")


    def model(x, w, b):
        return tf.add(x, w) + b


    y_model = model(X, w, b)

with tf.name_scope("CostFunction"):
    # use square error for cost func
    cost = tf.reduce_mean(tf.pow(Y - y_model, 2))

train_op = tf.train.AdamOptimizer(0.001).minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
# tf.summary.FileWriter(sess.graph, '/home/abondar/IdeaProjects/MachineLearnPython', 'graph.pbtxt')
cost_op = tf.summary.scalar("loss", cost)
merged = tf.summary.merge_all()
sess.run(init)
# writer = tf.summary.FileWriter('/home/abondar/IdeaProjects/MachineLearnPython', sess.graph)

x_vals = df[[df.columns[2], df.columns[4]]].values.astype(float)
y_vals = df[df.columns[12]].values.astype(float)

b0_temp = b.eval(session=sess)
b1_temp = w.eval(session=sess)

for a in range(1, 10):
    cost_1 = 0.0
    for i, j in zip(x_vals, y_vals):
        sess.run(train_op, feed_dict={X: i, Y: j})
        cost_1 += sess.run(cost, feed_dict={X: i, Y: j})/506.00
    x_vals, y_vals = shuffle(x_vals,y_vals)
    print(cost_1)
    b0_temp = b.eval(session=sess)
    b1_temp = w.eval(session=sess)
    print(b0_temp)
    print(b1_temp)
