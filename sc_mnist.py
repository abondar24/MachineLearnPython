import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from sc_mnist_nnet import NeuralNetMLP

# load and unpack mnist ds before running


def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


x_train, y_train = load_mnist('mnist', kind='train')
print('Rows: %d, columns: %d' % (x_train.shape[0], x_train.shape[1]))

x_test, y_test = load_mnist('mnist', kind='t10k')
print('Rows: %d, columns: %d' % (x_test.shape[0], x_test.shape[1]))

# show 10 examples
fix, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
    img = x_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# show 25 examples of 4
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(25):
    img = x_train[y_train == 4][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# 784 inp units, 50 hidden units, 10 output
# epochs - passes through training set
# eta - learning rate
# alpha - momentum learning param
# decrease_const - adaptive learning rate
# minibatches - we split data into k minibatches on every epoch
# for gradient bean calculated separately for every minibatch
# to make faster computation
nn = NeuralNetMLP(n_output=10, n_features=x_train.shape[1], n_hidden=50,
                  l2=0.1, l1=0.0, epochs=1000, eta=0.001, alpha=0.001,
                  decrease_const=0.00001, shuffle=True, minibatches=50, random_state=1)

nn.fit(x_train, y_train, print_progress=True)

plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([0, 2000])
plt.ylabel('Cost')
plt.xlabel('Epochs * 50')
plt.tight_layout()
plt.show()

# better picture after 800 epochs
batches = np.array_split(range(len(nn.cost_)), 1000)
cost_ary = np.array(nn.cost_)
cost_avgs = [np.mean(cost_ary[i]) for i in batches]

plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
plt.ylim([0, 2000])
plt.ylabel('Cost')
plt.xlabel('Epochs ')
plt.tight_layout()
plt.show()

y_train_pred = nn.predict(x_train)
acc = np.sum(y_train == y_train_pred, axis=0) / x_train.shape[0]
print('Training accuracy: %.2f%%' % (acc * 100))

y_test_pred = nn.predict(x_test)
acc = np.sum(y_test == y_test_pred, axis=0) / x_test.shape[0]
print('Training accuracy: %.2f%%' % (acc * 100))

miscl_img = x_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test != y_test_pred][:25]

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title(' %d) t: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i]))

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
