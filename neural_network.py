#!/usr/bin/python

import numpy

from keras.models import Sequential

model = Sequential()
model.add(Dense(10, input_shape = (28*28, ), kernel_initializer='he_normal'))
model.add(Activation('relu'))