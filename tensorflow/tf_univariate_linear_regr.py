import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# create linear space of 101 points between -1 and 1
tr_x = np.linspace(-1, 1, 101)

# create y function based on the x axis
tr_y = 2 * tr_x + np.random.randn(*tr_x.shape) * 0.4 + 0.2

plt.scatter(tr_x, tr_y)
plt.plot(tr_x, .2 + 2 * tr_x)
plt.show()

# create symbolic variables
X = tf.placeholder("float", name="X")
Y = tf.placeholder("float", name="Y")

# define a model
with tf.name_scope("Model"):
    def model(x, w, b):
        # define the line x*w+b
        return tf.add(x, w) + b


    # create shared variables
    w = tf.Variable(-1.0, name="b0")
    b = tf.Variable(-2.0, name="b1")

    y_model = model(X, w, b)

# cost function optimization
with tf.name_scope("CostFunction"):
    # use square error for cost func
    cost = (tf.pow(Y - y_model, 2))

train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
# tf.summary.FileWriter(sess.graph, '/home/abondar/IdeaProjects/MachineLearnPython', 'graph.pbtxt')
cost_op = tf.summary.scalar("loss", cost)
merged = tf.summary.merge_all()
sess.run(init)

# writer = tf.summary.FileWriter('/home/abondar/IdeaProjects/MachineLearnPython', sess.graph)

for i in range(100):
    for (x, y) in zip(tr_x, tr_y):
        sess.run(train_op, feed_dict={X: x, Y: y})
        summary_str = sess.run(cost_op, feed_dict={X: x, Y: y})
        # writer.add_summary(summary_str, i)

    b0_temp = b.eval(session=sess)
    b1_temp = w.eval(session=sess)


print(sess.run(w))
print(sess.run(b))

plt.scatter(tr_x, tr_y)
plt.plot(tr_x, sess.run(b) + tr_x * sess.run(w))
plt.show()
