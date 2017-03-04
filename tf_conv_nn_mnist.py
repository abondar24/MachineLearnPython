import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("tmp/data", one_hot=True)

# params
learning_rate = 0.001
training_iters = 2000
batch_size = 120
display_step = 10

# network params

# MNIST data input (28*28)
n_input = 784

# MNIST total classes (0-9)
n_classes = 10

# dropout(probability to keep units)
dropout = 0.8

# tf graph input
X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_classes])

keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W, b, strides=1):
    # conv2d layer with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding="SAME")
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # maxpool2d wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")


# create model
def conv_net(x, weights, biases, dropout):
    # reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # conv layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # max pooling(down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # conv layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # max pooling
    conv2 = maxpool2d(conv2, k=2)

    # fully connected layer
    # reshape to fit fully connected input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # apply dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# store layers weight and bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),

    # 5x5 cov, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),

    # fully connected, 7*7*64 inputs , 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),

    # 1024 inputs, 10 outputs(prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# contruct model
pred = conv_net(X, weights, biases, keep_prob)

# define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels= Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# init all variables
init = tf.global_variables_initializer()

# launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        test = batch_x[0]
        fig = plt.figure()
        plt.imshow(test.reshape((28, 28), order='C'), cmap='Greys', interpolation='nearest')
        plt.show()
        print(weights['wc1'].eval()[0])


        # run optimization op(backprop)
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
        if step % display_step == 0:
            # calc batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss) + ", Training Accuracy= " +
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization finished")
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={
        X: mnist.test.images[:256],
        Y: mnist.test.labels[:256],
        keep_prob: 1.
    }))
