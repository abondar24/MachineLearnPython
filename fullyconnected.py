from __future__ import print_function

import numpy as np
import tensorflow as tf
import pickle

image_size = 28
num_labels = 10

# Reformat data as a flat matrix
# Reformat labels as foat 1-hot encodings
def reformant(dataset,labels):
    dataset = dataset.reshape((-1,image_size * image_size)).astype(np.float32)
    # mapinng 0 to [1,0,0] and 1 to [0,1,0]
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

# Logistic regression
train_subset = 10000
graph = tf.Graph()
with graph.as_default():
    # Input data
    # Load everything to constants which are attached to graph
    tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
    tf_train_labels = tf.constant(train_labels[:train_subset])
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Parameters to train.
    # Weight matrix is initialized using random vals following a normal distribution
    # Biases are zero
    weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    # Training computation
    # weights*input + biases
    # Softmax and Cross-entropy are computed
    # We need the avg of cross-entropy
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer
    # Find min of Loss using gradient descent
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

num_steps = 701

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Logistic regression Initialized')
    for step in range(num_steps):
        # Run computations
        _, l, predictions = session.run([optimizer, loss, train_prediction])
        if step % 100 == 0:
            print('Loss at step %d: %f' % (step, l))
            print('Training accuracy: %1.f%%' % accuracy(
                predictions, train_labels[:train_subset, :]))

            # Run for val_prediction. It recomputes all its graph dependencies
            print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
