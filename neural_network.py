#!/usr/bin/python

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from matplotlib import pyplot as plt


#########print the first element of images.npy
img = np.load('images.npy')
img_flat = img.reshape(6500, 784)  # flatten matrix
labels = np.load('labels.npy')
print ("shape", img_flat.shape)
print("Number is ",labels[60])
plt.imshow(img[60])
plt.show()

########preprocessor variables (x_train, y_train ,etc)
x_train = img_flat[4224:0] #65% for training
x_val = img_flat[5199:4225] #15% for validation

y_train = labels[4224:0]
y_val =  labels [5199:4225]

########Create Model
model = Sequential()

########Define input layer
model.add(Dense(10, input_shape = (28*28, ), kernel_initializer='he_normal'))
model.add(Activation('relu'))
#
#
#
# Fill in Model Here
#
#

#########Define last layer
model.add(Dense(10, kernel_initializer='he_normal'))
model.add(Activation('softmax'))

#########Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#########Train Model
history = model.fit(x_train, y_train,
                    validation_data = (x_val, y_val),
                    epochs=10,
                    batch_size=512)

#########Report Results
print (history.history)
model.predict()