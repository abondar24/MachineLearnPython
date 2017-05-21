import numpy as np
import tensorflow as tf
import pickle


image_size = 28
num_labels = 10

# grayscale
num_channels = 1

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64


# Reformat data as a cube
# Reformat labels as float 1-hot encodings
def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


def accuracy(preds, labels):
    return 100.0 * np.sum(np.argmax(preds, 1) == np.argmax(labels, 1)) / preds.shape[0]


def model(data):
    conv = tf.nn.conv2d(data, weights1, [1, 1, 1, 1], padding='SAME')
    bias = tf.nn.relu(conv + biases1)
    pool = tf.nn.max_pool(bias, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    conv = tf.nn.conv2d(pool, weights2, [1, 1, 1, 1], padding='SAME')
    bias = tf.nn.relu(conv + biases2)
    pool = tf.nn.max_pool(bias, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    shape = pool.get_shape().as_list()
    reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, weights3) + biases3)
    return tf.matmul(hidden, weights4) + biases4


pickle_file = 'notMNIST.pickle'

with open(pickle_file,'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save

    print('Training set:', train_dataset.shape, train_labels.shape)
    print('Validation: set', valid_dataset.shape, valid_labels.shape)
    print('Testing set:', test_dataset.shape, test_labels.shape)

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('After reformat')
print('Training set:', train_dataset.shape, train_labels.shape)
print('Validation: set', valid_dataset.shape, valid_labels.shape)
print('Testing set:', test_dataset.shape, test_labels.shape)


# Convolution network with two layers followed by one fully connected layer(using pooling of stride 2 and kernel size 2)
graph = tf.Graph()
with graph.as_default():
    # Input data
    # Load everything to constants which are attached to graph
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Parameters to train.
    weights1 = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    biases1 = tf.Variable(tf.zeros([depth]))

    weights2 = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    biases2 = tf.Variable(tf.constant(1.0, shape=[depth]))

    weights3 = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    biases3 = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

    weights4 = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    biases4 = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    # Training computation
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=logits))

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

num_steps = 1001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Convolution neural network with pooling initialized')
    for step in range(num_steps):
        # Pick offset within randomized train data
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        # Dict with node of graph to be fed as key and numpy array to feed as value
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if step % 50 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            # Run for val_prediction. It recomputes all its graph dependencies
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
    acc = accuracy(test_prediction.eval(), test_labels)
    print('Test accuracy: %.1f%%' % acc)
