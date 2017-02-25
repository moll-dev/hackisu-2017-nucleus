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

data_path = 'C:\\scratch'
img_path = 'E:\\Users\\micha\\OneDrive\\Documents\\School\\Undergrad\\Research\\corn\\CAE\\test_field_images'


patch_size = 256
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
test_img = img_path + '\\DSCN1853.JPG'
patcher_test = Patcher.from_image(test_img, None, _dim=(patch_size, patch_size), _stride=(64, 64))
print(patcher_test.img_arr.shape)
print('Done.')

def predictor(_patches):
    return autoencoder.predict(_patches)

pred_label = patcher_test.predict(predictor, frac=0.33)
pred_label = pred_label / 9.0
pred_label = np.clip(pred_label, 0.0, 1.0)

plt.hist(pred_label.flatten(), 50)
plt.show()

#pred_label = np.where(pred_label > 9.0, 9)
#pred_label = np.rint(pred_label-0.49)

img = Image.fromarray(pred_label)
img = img.convert('L')
img = img.filter(ImageFilter.BLUR)

plt.imshow(pred_label)
plt.show() 