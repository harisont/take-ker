# A stacked autoencoder of MNIST digits based on a Keras blog tutorial,
# available at: https://blog.keras.io/building-autoencoders-in-keras.html
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.utils import print_summary


print("LOADING DATA...")
# the second element of the tuples is not relevant, cause the labels are not
# required for unsupervised learning
(x_train, _), (x_test, _) = mnist.load_data()

print("PREPROCESSING INPUT DATA...")
# convert data type to float32 and normalize so that they are btw 0 and 1
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# flatten the 28x28 images into vectors of size 784
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print("SETTING UP NETWORK ARCHITECTURE...")
# size of the encoded representation (number of floats representing it)
# 32 means that the compression factor is 24.5 (1/4), as the input is 784
encoding_dim = 32  

# input layer
inputs = Input(shape=(784,))

# encoder
encoded = Dense(128, activation='relu')(inputs)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

# decoder
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

# creation of three separate models for the whole autoencoder,
# the encoder and the decoder, so that they can be then used separately
autoencoder = Model(inputs, decoded)

encoder = Model(inputs, encoded)

decoder_input = Input(shape=(encoding_dim,))
decoder_output = autoencoder.layers[-3](decoder_input)
decoder_output = autoencoder.layers[-2](decoder_output)
decoder_output = autoencoder.layers[-1](decoder_output)
decoder = Model(decoder_input, decoder_output)

print_summary(autoencoder)

print("COMPILING MODEL...")
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

print("TRAINING MODEL...")
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

print("TESTING MODEL...")
# encode and decode some digits from the test set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

print("PLOTTING A BUNCH OF RESULTS...")
n = 10  # number og digits to be displayed
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()