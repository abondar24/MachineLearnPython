import theano
import os
import struct
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

# on windows set default keras backend as theano
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

theano.config.floatX = 'float32'
x_train = x_train.astype(theano.config.floatX)
x_test = x_test.astype(theano.config.floatX)

print('First 7 labels: ', y_train[:7])
y_train_ohe = np_utils.to_categorical(y_train)
print('\nFirst 7 labels(one-hot):\n', y_train[:7])

np.random.seed(1)
model = Sequential()

#input layer
model.add(Dense(input_dim=x_train.shape[1],
                output_dim=50,
                init='uniform',
                activation='tanh'))

model.add(Dense(input_dim=50,
                output_dim=50,
                init='uniform',
                activation='tanh'))

model.add(Dense(input_dim=50,
                output_dim=y_train_ohe.shape[1],
                init='uniform',
                activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])

model.fit(x_train, y_train_ohe, nb_epoch=50, batch_size=300,
          verbose=1, validation_split=0.1)

y_train_predict = model.predict_classes(x_train, verbose=0)
print('First 7 predictions: ', y_train_predict[:7])
train_acc = np.sum(y_train == y_train_predict, axis=0) / x_train.shape[0]
print('Training accuracy: %.2f%%' % (train_acc * 100))

y_test_predict = model.predict_classes(x_test, verbose=0)
print('First 7 predictions: ', y_test_predict[:7])
test_acc = np.sum(y_test == y_test_predict, axis=0) / x_test.shape[0]
print('Test accuracy: %.2f%%' % (test_acc * 100))

