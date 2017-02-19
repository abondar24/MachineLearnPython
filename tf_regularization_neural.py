from __future__ import print_function

import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

image_size = 28
num_labels = 10

# Reformat data as a flat matrix
# Reformat labels as float 1-hot encodings
def reformant(dataset,labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


def accuracy(preds, labels):
    return 100.0 * np.sum(np.argmax(preds, 1) == np.argmax(labels, 1)) / preds.shape[0]

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

train_dataset, train_labels = reformant(train_dataset, train_labels)
valid_dataset, valid_labels = reformant(valid_dataset, valid_labels)
test_dataset, test_labels = reformant(test_dataset, test_labels)
print('After reformat')
print('Training set:', train_dataset.shape, train_labels.shape)
print('Validation: set', valid_dataset.shape, valid_labels.shape)
print('Testing set:', test_dataset.shape, test_labels.shape)


# 1-Layer neuron network with regularization
num_hidden_nodes = 1024
batch_size = 128
graph = tf.Graph()
with graph.as_default():
    # Input data
    # Load everything to constants which are attached to graph
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    beta_regul = tf.placeholder(tf.float32)

    # Parameters to train.
    weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_hidden_nodes]))
    biases1 = tf.Variable(tf.zeros([num_hidden_nodes]))

    weights2 = tf.Variable(tf.truncated_normal([num_hidden_nodes, num_labels]))
    biases2 = tf.Variable(tf.zeros([num_labels]))

    # Training computation
    lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
    logits = tf.matmul(lay1_train, weights2) + biases2
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + \
           beta_regul * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2))


    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    train_prediction = tf.nn.softmax(logits)
    lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
    valid_prediction = tf.nn.softmax(tf.matmul(lay1_valid, weights2) + biases2)
    lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
    test_prediction = tf.nn.softmax(tf.matmul(lay1_test, weights2) + biases2)

num_steps = 3001

# single run with param val 1e-3
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    for step in range(num_steps):
        # Pick offset within randomized train data
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        # Dict with node of graph to be fed as key and numpy array to feed as value
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, beta_regul: 1e-3}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    acc = accuracy(test_prediction.eval(), test_labels)
    print('Test accuracy: %.1f%%' % acc)

# We make a plot to see accuracies on different meta param values
# Clear cache rm /home/deeplearn/.cache/matplotlib/fontList*.cache in case of error
regul_val = [pow(10, i) for i in np.arange(-4, -2, 0.1)]
accuracy_val = []

for regul in regul_val:
   with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        for step in range(num_steps):
            # Pick offset within randomized train data
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            # Dict with node of graph to be fed as key and numpy array to feed as value
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, beta_regul: regul}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if step % 500 == 0:
                print("Minibatch loss at step %d: %f" % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                # Run for val_prediction. It recomputes all its graph dependencies
                print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
        acc = accuracy(test_prediction.eval(), test_labels)
        print('Test accuracy: %.1f%%' % acc)
        accuracy_val.append(acc)

plt.semilogx(regul_val, accuracy_val)
plt.grid(True)
plt.title('Test accuracy by regularization(1-layer net)')
plt.show()


