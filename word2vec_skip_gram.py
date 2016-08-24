from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile

from matplotlib import pylab
from urllib.request import urlretrieve
from sklearn.manifold import TSNE

# global variables
url = 'http://mattmahoney.net/dc/'
vocabulary_size = 50000
data_index = 0


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
    """Extract the first file enclosed in a zip as a lift of words"""
    with zipfile.ZipFile(fname) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


# build the dictionary and replace rare words with UNK token
def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            # in UNK dict
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dict = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dict


# generate a training batch for skip-gram model
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=batch_size, dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # [skip_window target skip_window]
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        # target label at the center of the buffer
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


def plot(embeds, lbels):
    assert embeds.shape[0] >= len(lbels), 'More labels than embeddings'
    pylab.figure(figsize=(15, 15))
    for i, label in enumerate(lbels):
        x, y = embeds[i, :]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    pylab.show()


filename = download('text8.zip', 31344016)
words = read_data(filename)
print('Data size %d' % len(words))

data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:10])
print('Sample data', data[:10])
# reduce memory
del words

print('data:', [reverse_dictionary[di] for di in data[:32]])
for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch(batch_size=16, num_skips=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(16)])

for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 1
    batch, labels = generate_batch(batch_size=16, num_skips=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(16)])

print(batch)
print(labels)

# train a skip-gram model
batch_size =128

# dimensions of the embedding vector
embedding_size = 128

# how many words to consider left and right
skip_window = 1

# how many times to reuse an input to generate a label
num_skips = 2

# pick a random validation set to sample nearest neighbours.
# samples are limited to the words that have a low numeric id
# and which are the most frequent by construction

# random set of words to evaluate similarity on
valid_size = 16

# only pick dev samples in the head of the distribution
valid_window = 100

valid_examples = np.array(random.sample(range(valid_window), valid_size))

# number of negative examples to samples
num_sampled = 64

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):

    # input data
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # variables
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                      stddev=1.0 / math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # model
    # look up embeddings for inputs
    embeds = tf.nn.embedding_lookup(embeddings, train_dataset)

    # compute the softmax loss, using a sample of the negative labels each time
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embeds,
                                                     train_labels, num_sampled, vocabulary_size))

    # optimizer
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    # compute the similarity between minibatch examples and all embedings using cosine distance
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

num_steps = 100001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initilized')
    average_loss = 0
    for step in range(num_steps):
        batch_data, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # avg loss is an estimate of the loss over the last 2000 batches
            print('Average loss at step %d: %f' % (step, average_loss))
            average_loss = 0

        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                # number of nearest neighbours
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s, ' % (log, close_word)
                print(log)
    final_embeddings = normalized_embeddings.eval()

print(final_embeddings[0])

# if 1.0 than embeds are normalized
print(np.sum(np.square(final_embeddings[0])))

num_points = 400
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])
words = [reverse_dictionary[i] for i in range(1, num_points + 1)]
plot(two_d_embeddings, words)