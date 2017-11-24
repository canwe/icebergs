import json
import sys
from PIL import Image

import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping


np.random.seed(25)

""" load the training data """
# list of dictionnaries with keys ('id', 'band_1', 'band_2', 'inc_angle', 'is_iceberg')
data_parsed = json.loads(open('train.json').read())

X1 = []
X2 = []
y = []

for image in data_parsed:
    is_iceberg = image['is_iceberg']
    
    data = image['band_1']
#    for i in range(len(data)):
#        data[i] = image['band_1'][i]/image['band_2'][i]

    min_data = np.min(data)
    max_data = np.max(data)
    for i in range(len(data)):
        data[i] = (data[i]-min_data)/(max_data-min_data)
    
    X1.append(data)

    data = image['band_2']

    min_data = np.min(data)
    max_data = np.max(data)
    for i in range(len(data)):
        data[i] = (data[i]-min_data)/(max_data-min_data)
    
    X2.append(data)

    y.append(is_iceberg)

# use also some clearcut cases from scored_test.json

data_parsed = json.loads(open('scored_test.json').read())

for image in data_parsed:
    is_iceberg = image['is_iceberg']
    if is_iceberg>0.9:
        is_iceberg = 1
    elif is_iceberg<0.1:
        is_iceberg = 0
    else:
        continue
    
    data = image['band_1']
    #    for i in range(len(data)):
    #        data[i] = image['band_1'][i]/image['band_2'][i]
    
    min_data = np.min(data)
    max_data = np.max(data)
    for i in range(len(data)):
        data[i] = (data[i]-min_data)/(max_data-min_data)

    X1.append(data)

    data = image['band_2']

    min_data = np.min(data)
    max_data = np.max(data)
    for i in range(len(data)):
        data[i] = (data[i]-min_data)/(max_data-min_data)
    
    X2.append(data)
    y.append(is_iceberg)

X1 = np.array(X1)
X2 = np.array(X2)
X = np.concatenate((X1[...,np.newaxis],X2[...,np.newaxis]),axis=-1)


X_train, X_test, y_train, y_test = train_test_split(X[:], y[:], test_size=0.33)
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = np.array(X_train)
X_test = np.array(X_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)
print("X_test original shape", X_test.shape)
print("y_test original shape", y_test.shape)



X_train = X_train.reshape(X_train.shape[0], 75, 75, 2)
X_test = X_test.reshape(X_test.shape[0], 75, 75, 2)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')



print(X_train.shape)

number_of_classes = 2

Y_train = np_utils.to_categorical(y_train, number_of_classes)
Y_test = np_utils.to_categorical(y_test, number_of_classes)

print(y_train[0], Y_train[0])



# Three steps to Convolution
# 1. Convolution
# 2. Activation
# 3. Polling
# Repeat Steps 1,2,3 for adding more hidden layers

# 4. After that make a fully connected network
# This fully connected network gives ability to the CNN
# to classify the samples

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(75,75,2)))
model.add(Activation('relu'))
BatchNormalization(axis=-1)
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

BatchNormalization(axis=-1)
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
BatchNormalization(axis=-1)
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
# Fully connected layer

BatchNormalization()
model.add(Dense(512))
model.add(Activation('relu'))
BatchNormalization()
#model.add(Dropout(0.2))  # !!!
model.add(Dense(2))

# model.add(Convolution2D(10,3,3, border_mode='same'))
# model.add(GlobalAveragePooling2D())
model.add(Activation('softmax'))  # !!!
#model.add(Activation('sigmoid'))  # !!!

# load weights
model.load_weights("weights.best.hdf5")

model.summary()
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)

test_gen = ImageDataGenerator()

train_generator = gen.flow(X_train, Y_train, batch_size=64)
test_generator = test_gen.flow(X_test, Y_test, batch_size=64)

# model.fit(X_train, Y_train, batch_size=128, nb_epoch=1, validation_data=(X_test, Y_test))

model.fit_generator(train_generator, steps_per_epoch=1074//64, epochs=5,
                    validation_data=test_generator, validation_steps=530//64, callbacks=callbacks_list)

score = model.evaluate(X_test, Y_test)
print()
print('Test accuracy: ', score[1])

predictions = model.predict_classes(X_test)

predictions = list(predictions)
actuals = list(y_test)

sub = pd.DataFrame({'Actual': actuals, 'Predictions': predictions})
sub.to_csv('./output_cnn.csv', index=False)


if score[1]<0.7:
    print "Test accuracy was insufficient, not going to evaluate actual unknown data."
    quit()
print("Continuing with predictions for real data..")

""" load the data with unknown answer """
print("Loading the data with unknown answers..")
data_parsed = json.loads(open('test.json').read())

X_unknown1 = []
X_unknown2 = []
ids = []

for image in data_parsed[:]:
    id = image['id']
    
    data = image['band_1']
    min_data = np.min(data)
    max_data = np.max(data)
    for i in range(len(data)):
        data[i] = (data[i]-min_data)/(max_data-min_data)
    X_unknown1.append(data)

    data = image['band_2']
    min_data = np.min(data)
    max_data = np.max(data)
    for i in range(len(data)):
        data[i] = (data[i]-min_data)/(max_data-min_data)
    X_unknown2.append(data)

    ids.append(id)
X_unknown1 = np.array(X_unknown1)
X_unknown2 = np.array(X_unknown2)
X_unknown = np.concatenate((X_unknown1[...,np.newaxis],X_unknown2[...,np.newaxis]),axis=-1)
X_unknown = np.array(X_unknown)
X_unknown = X_unknown.reshape(X_unknown.shape[0], 75, 75, 2)

# predict the likelihood of being an iceberg
#predictions = model.predict_classes(X_unknown)
predicts = model.predict(X_unknown,verbose=0)

predictions = []
for line in predicts:
    predictions.append(line[1]) # take only probability that is_iceberg = True

sub = pd.DataFrame({'id': ids, 'is_iceberg': predictions})
sub.to_csv('./output_unknown_cnn.csv', index=False)



class MixIterator(object):
    def __init__(self, iters):
        self.iters = iters
        self.N = sum([it.n for it in self.iters])
    
    def reset(self):
        for it in self.iters: it.reset()
    
    def __iter__(self):
        return self
    
    def __next__(self, *args, **kwargs):
        nexts = [next(it) for it in self.iters]
        n0 = np.concatenate([n[0] for n in nexts])
        n1 = np.concatenate([n[1] for n in nexts])
        return (n0, n1)

predictions = model.predict(X_test, batch_size=64)

predictions[:5]

# gen = ImageDataGenerator()

batches = gen.flow(X_train, Y_train, batch_size=48)
test_batches = test_gen.flow(X_test, predictions, batch_size=16)

mi = MixIterator([batches, test_batches])

mi.N

#model.fit_generator(mi, steps_per_epoch=mi.N//64, epochs=5, validation_data=(X_test, Y_test))
# the last thing didn't work

