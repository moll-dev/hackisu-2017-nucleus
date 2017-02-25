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
from PIL import ImageFilter, Image
import scipy.misc as misc

data_path = 'C:\\Users\\micha\\OneDrive\\Documents\\hackisu2017'
img_path = 'C:\\Users\\micha\\Pictures\\Kinect_Capture_005450'


patch_size = 200
input_img = Input(shape=(patch_size, patch_size, 3))

x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

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

autoencoder.load_weights(data_path + '\\cae_weights.h5')

print('Creating test patcher with dim=(32, 32).')
test_img = img_path + '\\capture736c.jpg'
patcher_test = Patcher.from_image(test_img, None, _dim=(patch_size, patch_size), _stride=(64, 64))

print(patcher_test.img_arr.shape)
print('Done.')

def predictor(_patches):
    return autoencoder.predict(_patches)

pred_label = patcher_test.predict(predictor, frac=1.0)
pred_label = pred_label
pred_label = np.clip(pred_label, 0.0, 1.0)

plt.hist(pred_label.flatten(), 50)
plt.show()

#pred_label = np.where(pred_label > 9.0, 9)
#pred_label = np.rint(pred_label-0.49)

misc.imsave('test.png', pred_label)

plt.imshow(pred_label)
plt.show() 