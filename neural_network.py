#!/usr/bin/python

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
import pandas as pd
from matplotlib import pyplot as plt

np.random.seed(117)

#########print the first element of images.npy
img = np.load('images.npy')
img_flat = img.reshape(6500, 784)  # flatten matrix
labels = np.load('labels.npy')
labels_flat = labels.reshape(6500 ,1)
print("Number is ",labels[6400])
#plt.imshow(img[6400])
#plt.show()

########preprocessor variables (x_train, y_train ,etc)
x_train = img_flat[0:4224] #65% for training
x_val = img_flat[4225:5119] #15% for validation

y_train = np_utils.to_categorical(labels[0:4224], 10)
y_val = np_utils.to_categorical(labels[4225:5119], 10)

# print ("y_train[0]", y_train[0])
# print ("labels[0]", labels[0])

########Create Model
model = Sequential()

########Define input layer
model.add(Dense(50, input_shape = (28*28, ), kernel_initializer='he_normal'))
model.add(Activation('relu'))

#######Hidden Layers
model.add(Dense(50, kernel_initializer = 'he_normal'))
model.add(Activation('relu'))


model.add(Dense(50, kernel_initializer='he_normal'))
model.add(Activation('relu'))

model.add(Dense(60, kernel_initializer='he_normal'))
model.add(Activation('relu'))

model.add(Dense(60, kernel_initializer='he_normal'))
model.add(Activation('relu'))


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
                    epochs=40,
                    batch_size=512)

#########Report Results
print (history.history)
prediction = model.predict(img_flat[6400:6499])#, batch_size=10, verbose=1)
actual = labels[6400:6499]

########Print results
for result in range(len(prediction)):
    for num in range(len(prediction[result])):
        if prediction[result][num] == max(prediction[result]):
            print ("Actual :", actual[result], end="")
            print ("  Prediction :", num)
