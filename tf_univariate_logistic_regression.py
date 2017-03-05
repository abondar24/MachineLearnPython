import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

df = pd.read_csv("chd.csv", header=0)
print(df.describe())

learning_rate = 0.2
batch_size = 100
display_step = 1
training_epochs = 5

sess = tf.Session()
b = np.zeros((100, 2))

# placeholder for 1D data
x = tf.placeholder("float", [None, 1])

# placeholder for classes
y = tf.placeholder("float", [None, 2])

# linear model vars
W = tf.Variable(tf.zeros([1, 2]))
b = tf.Variable(tf.zeros([2]))

# activation func
activation = tf.nn.softmax(tf.matmul(x, W) + b)

# cross entropy for error minimize
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(activation), reduction_indices=1))

# croos entropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    tf.train.write_graph(sess.graph, './graphs', 'graph.pbtxt')
    sess.run(init)
    writer = tf.summary.FileWriter('./graphs', sess.graph)

    # init graph struct

    graph_number = 321
    plt.figure(1)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(400 / batch_size)

        for i in range(total_batch):
            temp = tf.one_hot(indices=df['chd'].values, depth=2, on_value=1,
                              off_value=0, axis=-1, name="a")

            batch_xs, batch_ys = (np.transpose([df['age']]) - 44.38) / 11.721327, temp

            # fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs.astype(float), y: batch_ys.eval()})

            # compute avg loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs.astype(float),
                                                  y: batch_ys.eval()}) / total_batch

        if epoch % display_step == 0:
            print("Epoch:", '%05d' % (epoch + 1),
                  "cost=", "{:.8f}".format(avg_cost))

            tr_X = np.linspace(-30, 30, 100)
            print(b.eval())
            print(W.eval())
            W_dos = 2 * W.eval()[0][0] / 11.721327
            b_dos = 2 * b.eval()[0]

            # gen probability func
            tr_y = np.exp(-((W_dos * tr_X) + b_dos) / (1 + np.exp(-(W_dos * tr_X) + b_dos)))

            # draw samples and prob func
            plt.subplot(graph_number)
            graph_number = graph_number + 1

            # plot random datapoints
            plt.scatter((df['age']), df['chd'])
            plt.plot(tr_X + 44.38, tr_y)
            plt.grid(True)

        plt.savefig("save.svg")
