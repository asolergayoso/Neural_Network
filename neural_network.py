#!/usr/bin/python

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from matplotlib import pyplot as plt

np.random.seed(117) #seeding to improve consistency

#########print the first element of images.npy
img = np.load('images.npy')
img_flat = img.reshape(6500, 784)  # flatten matrix
labels = np.load('labels.npy')
labels_flat = labels.reshape(6500 ,1)

########preprocessor variables (x_train, y_train ,etc)
x_train = img_flat[0:3899] #60% for training
x_val = img_flat[3900:4874] #15% for validation

y_train = np_utils.to_categorical(labels[0:3899], 10) #change the fromat to arrays of 1's and 0's
y_val = np_utils.to_categorical(labels[3900:4874], 10)

########Create Model
model = Sequential()

########Define input layer and second layer
model.add(Dense(50, input_shape = (28*28, ), kernel_initializer='random_uniform'))
model.add(Activation('relu'))

#######Hidden Layers
model.add(Dense(50, kernel_initializer = 'he_normal'))
model.add(Activation('relu'))

model.add(Dense(50, kernel_initializer='he_normal'))
model.add(Activation('relu'))

model.add(Dense(60, kernel_initializer='random_uniform'))
model.add(Activation('relu'))

model.add(Dense(60, kernel_initializer='random_uniform'))
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
prediction = model.predict(img_flat[4875:6499], verbose = 1)
actual = labels[4875:6499]

########Print results
prediction_nums = []
for result in range(len(prediction)):
    for num in range(len(prediction[result])):
        if prediction[result][num] == max(prediction[result]):
            print ("Actual :", actual[result], "-->", end=" ")
            print ("Prediction :", num, "-->", end=" ")
            if (num == actual[result]):
                print ("True")
            else:
                print("False")
            prediction_nums.append(num)


#######Visualize three missclassified images
count = 1
for miss in range(len(prediction_nums)):
    if prediction_nums[miss] != actual[miss] and count <= 3:
        count += 1
        plt.imshow(img[4875 + miss], cmap='gray')
        plt.show()