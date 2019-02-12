# MNIST classification using Theano backend,
# based on EliteDataScience tutorial

from keras.datasets import mnist
from keras.utils import np_utils, plot_model
# simple linear stack of NN layers:
from keras.models import Sequential 
# "core" keras layers:
from keras.layers import Dense, Dropout, Activation, Flatten
# convolutional keras layers
from keras.layers import Convolution2D, MaxPooling2D
import numpy as np
from matplotlib import pyplot as plt

from keras import backend as K
K.set_image_dim_ordering('th')

np.random.seed(123) # to be able to reproduce results

# LOADING DATA
# load pre-shuffled MNIST data into train and test sets
# NB: x : input data = y : labels
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# USELESS PRINTING STUFF
# print the shape of the training and test set
(n,w,h) = x_train.shape
print ("Training set: {} {}x{} samples".format(n,w,h))
(n,w,h) = x_test.shape
print ("    Test set: {} {}x{} samples".format(n,w,h))

# plot one sample just to see how it looks like
plt.imshow(x_train[0])
# plt.show()

# PREPROCESSING INPUT DATA
# reshape data: (n, width, height) -> (n, depth, width, height)
# (NB: this is Theano backend specific)
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
x_test  = x_test.reshape(y_test.shape[0], 1, 28, 28)

# convert data type to float32
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')

# normalize
x_train /= 255
x_test  /= 255

# PREPROCESSING CLASS LABELS
# y_train.shape (and y_test.shape) should be 10 different classes (one per
# digit), but they are actually encoded just as a 1-dimensional array, so:
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# DEFINING NETWORK ARCHITECTURE
model = Sequential()

# input layer
# 32: number of convolution filters
#  3: numbers of rows (and columns) per convolution kernel
model.add(Convolution2D(32, (3,3), activation='relu', input_shape=(1,28,28)))

# some other layers
model.add(Convolution2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2))) # to reduce the numebr of parameters
model.add(Dropout(0.25)) # to avoid overfitting

model.add(Flatten())
model.add(Dense(128, activation='relu')) #
model.add(Dropout(0.5))

# output layer (10: number of output neurons)
model.add(Dense(10, activation='softmax')) 

# COMPILING
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# plot the network
plot_model(model, to_file='model.png')

# TRAINING
model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)

# EVALUATING
score = model.evaluate(x_test, y_test, verbose=0)
