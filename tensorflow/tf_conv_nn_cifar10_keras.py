import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from tensorflow.contrib import learn
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Convolution2D,MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.utils.np_utils import to_categorical

datadir = 'cifar-10-batches-bin/'
G = glob.glob(datadir + '*.bin')
A = np.fromfile(G[0], dtype=np.uint8).reshape([10000, 3073])
labels = to_categorical(A[:, 0])
images = A[:, 1:].reshape([10000, 3, 32, 32]).transpose(0, 2, 3, 1)
images_unroll = A[:, 1:]

model = Sequential()

model.add(Convolution2D(16,5,5,border_mode='valid',input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(16,5,5,border_mode='valid',input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

model.fit(images[:8000],labels[:8000],batch_size=100,nb_epoch=300,
          verbose=1,validation_data=(images[8000:],labels[8000:]))
score = model.evaluate(images[8000:],labels[8000:],verbose=0)
print('Test score:',score[0])
print('Test accuracy:',score[1])
