from __future__ import print_function

import numpy as np
import os
import random
import string
import tensorflow as tf
import zipfile

from urllib.request import urlretrieve

# global variables
url = 'http://mattmahoney.net/dc/'

# [a-z] + ' '
vocabulary_size = len(string.ascii_lowercase) + 1

first_letter = ord(string.ascii_lowercase[0])
batch_size = 64
num_unrollings = 10

def download(fname, expected_bytes):
    if not os.path.exists(fname):
        fname, _ = urlretrieve(url + fname, fname)

    stat_info = os.stat(fname)
    if stat_info.st_size == expected_bytes:
        print('Found and verified', fname)
    else:
        print(stat_info.st_size)
        raise Exception('Failed to verify ' + fname + '. Can you get to it with a browser?')
    return fname


def read_data(fname):
    f = zipfile.ZipFile(fname)
    for name in f.namelist():
        return tf.compat.as_str(f.read(name))
    f.close()


# util funcitons to map chars to vocabulary IDs and back
def char2id(char):
    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 1
    elif char == ' ':
        return 0
    else:
        print('Unexpected character: %s' % char)
        return 0


def id2char(dictid):
    if dictid > 0:
        return chr(dictid + first_letter - 1)
    else:
        return ' '



class BatchGenerator(object):
    def __init__(self, txt, batch_sz, num_unrolls):
        self._text = txt
        self._text_size = len(txt)
        self._batch_size = batch_sz
        self._num_unrollings = num_unrolls
        segment = self._text_size // batch_sz
        self._cursor = [offset * segment for offset in range(batch_sz)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single natch from the curent cursor pos in the data"""
        batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, char2id(self._text[self._cursor[b]])] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        """Gen the next array of a probability distribution over the possible chars
        back into its char representation"""
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches


def characters(probs):
    """Turn a 1-hot encoding over the possible chars into its char representation"""
    return [id2char(c) for c in np.argmax(probs, 1)]


def batches2string(batches):
    """Convert batches to string representation"""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b))]
    return s


def logprob(preds,lbels):
    """Log-probability of the true labels in a predicted batch"""
    preds[preds < 1e-10] = 1e-10
    return np.sum(np.multiply(lbels, -np.log(preds))) / lbels.shape[0]


def sample_distribution(distrib):
    """Sample one lement from a distribution assumed
    to be an array of normalized probalilities"""

    r = random.uniform(0, 1)
    s = 0
    for i in range(len(distrib)):
        s += distrib[i]
        if s >= r:
            return i
    return len(distrib) - 1


def sample(pred):
    """Turn a prediction into 1-hot encoded samples."""
    p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
    p[0, sample_distribution(pred[0])] = 1.0
    return p


def random_distribution():
    """generate a random column of probalities"""
    b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
    return b/np.sum(b, 1)[:, None]


def lstm_cell(i, o, state):
    """Create a LSTM cell without  connections between previous state and gates"""
    smatmul = tf.matmul(i, sx) + tf.matmul(o, sm) + sb
    smatmul_input, smatmul_forget, update, smatmul_output = tf.split(1, 4, smatmul)
    input_gate = tf.sigmoid(smatmul_input)
    forget_gate = tf.sigmoid(smatmul_forget)
    output_gate = tf.sigmoid(smatmul_output)
    state = forget_gate * state + input_gate * tf.tanh(update)
    return output_gate * tf.tanh(state), state


filename = download('text8.zip', 31344016)
text = read_data(filename)
print('Data size %d' % len(text))

# create a validation set
valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

# simple LSTM model with single matrix
num_nodes = 64

graph = tf.Graph()
with graph.as_default():

        # Parameters:
        #  Input gate: input, previous output and bias.
        ix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
        im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
        ib = tf.Variable(tf.zeros([1, num_nodes]))
        # Forget gate: input, previous output and bias.
        fx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
        fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
        fb = tf.Variable(tf.zeros([1, num_nodes]))
        # Memory cell: input, state and bias
        cx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
        cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
        cb = tf.Variable(tf.zeros([1, num_nodes]))
        # Output gate: input, previous output and bias
        ox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
        om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
        ob = tf.Variable(tf.zeros([1, num_nodes]))
        # Concatenate parameters
        sx = tf.concat(1, [ix, fx, cx, ox])
        sm = tf.concat(1, [im, fm, cm, om])
        sb = tf.concat(1, [ib, fb, cb, ob])
        # Variables saving state across unrollings
        saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
        saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
        # Classifier weights and biases
        w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
        b = tf.Variable(tf.zeros([vocabulary_size]))

        # Input data
        train_data = list()
        for _ in range(num_unrollings + 1):
            train_data.append(tf.placeholder(tf.float32, shape=[batch_size, vocabulary_size]))
        train_inputs = train_data[:num_unrollings]
        # labels are inputs shifted by one time step
        train_labels = train_data[1:]

        # Unrolled LSTM loop
        outputs = list()
        output = saved_output
        state = saved_state
        for i in train_inputs:
            output, state = lstm_cell(i, output, state)
            outputs.append(output)

        # State saving across unrollings.
        with tf.control_dependencies([saved_output.assign(output), saved_state.assign(state)]):
            # Classifier
            logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, train_labels)))

        # Optimizer
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
        optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

        # Predictions
        train_prediction = tf.nn.softmax(logits)

        # Sampling and validation eval: batch 1, no unrolling
        sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
        saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
        saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
        reset_sample_state = tf.group(
            saved_sample_output.assign(tf.zeros([1, num_nodes])),
            saved_sample_state.assign(tf.zeros([1, num_nodes])))
        sample_output, sample_state = lstm_cell(sample_input, saved_sample_output, saved_sample_state)
        with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                      saved_sample_state.assign(sample_state)]):
            sample_prediction =tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))


num_steps = 7001
summary_frequency = 100

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Simple LSTM network with single matrix initialized")
    mean_loss = 0
    for step in range(num_steps):
        batches = train_batches.next()
        feed_dict = dict()
        for i in range(num_unrollings + 1):
            feed_dict[train_data[i]] = batches[i]
        _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
        mean_loss += l
        if step % summary_frequency == 0:
            if step > 0:
                mean_loss /= summary_frequency
            # The mean loss is an estimate of the loss over the last few batches
            print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
            mean_loss = 0
            labels = np.concatenate(list(batches)[1:])
            print('Minibatch perplexity: %.2f' % float(np.exp(logprob(predictions, labels))))
            if step % (summary_frequency * 10) == 0:
               # Generate samples
               print('=' * 80)
               for _ in range(5):
                   feed = sample(random_distribution())
                   sentence = characters(feed)[0]
                   reset_sample_state.run()
                   for _ in range(79):
                       prediction = sample_prediction.eval({sample_input: feed})
                       feed = sample(prediction)
                       sentence += characters(feed)[0]
                   print(sentence)
               print('=' * 80)
               # Measure validation set perplexity
               reset_sample_state.run()
               valid_logprob = 0
               for _ in range(valid_size):
                   b = valid_batches.next()
                   predictions = sample_prediction.eval({sample_input: b[0]})
                   valid_logprob = valid_logprob + logprob(predictions, b[1])
               print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_size)))