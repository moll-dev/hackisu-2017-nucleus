from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import numpy as np
from patcher import Patcher
import os
import random
from sklearn.feature_extraction import image
import h5py

data_path = 'C:\\scratch'

patch_size = 256

input_img = Input(shape=(patch_size, patch_size, 3))

x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)

encoded = MaxPooling2D((2, 2), border_mode='same')(x)

x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)

print(autoencoder.summary())
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

print('Loading data...')
with h5py.File(data_path + '\\train_25_256.h5') as f:
    x_train = np.array(f['images'])
    y_train = np.array(f['labels'])

## Train the AE
print('Training...')
autoencoder.fit(x_train, y_train,
                nb_epoch=10,
                batch_size=32,
                shuffle=True)
                #callbacks=[TensorBoard(log_dir='.\\tensorboard')])

autoencoder.save_weights(data_path + '\\cae_weights.h5')
