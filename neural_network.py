#!/usr/bin/python

import numpy as np
#from keras.models import Sequential
#from keras.layers import Dense, Activation
from matplotlib import pyplot as plt
#import tensorflow as tf

img = np.load('images.npy')
print (img[1].shape)
#img2 = np.random.rand(28, 28, 1)
#img = img.transpose(1)
plt.imshow(img[0], cmap='gray')
plt.show()

# model = Sequential()a
# model.add(Dense(10, input_shape = (28*28, ), kernel_initializer='he_normal'))
# model.add(Activation('relu'))
# #
# #
# #
# # Fill in Model Here
# #
# #
# model.add(Dense(10, kernel_initializer='he_normal')) # last layer
# model.add(Activation('softmax'))
#
# #Compile Model
# model.compile(optimizer='sgd',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# #Train Model
# history = model.fit(x_train, y_train,
#                     validation_data = (x_val, y_val),
#                     epochs=10,
#                     batch_size=512)
#
# #Report Results
#
# print (history.history)
# model.predict()