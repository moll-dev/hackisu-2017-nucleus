import matplotlib.pyplot as plt
import numpy as np
from patcher import Patcher
import os
import random
from sklearn.feature_extraction import image
import h5py
import gc
import sys

#dataset = sys.argv[1]
savename = sys.argv[1]

#data_path = 'C:\\nucleus\\'  + dataset

data_path = 'C:\\Users\\micha\\Pictures\\Kinect_Capture_005450'
patch_size = 512

imgs = os.listdir(data_path)

x_train = None
x_train_tmp = []
y_train = None
y_train_tmp = []

first = True

# Patching data
for train_img in imgs:
    if train_img.endswith('d.jpg'):
        continue

    train_lbl = train_img.replace('c.jpg', 'd.jpg')

    print('{} -> {}'.format(train_img, train_lbl))

    train_img = data_path + '\\' + train_img
    train_lbl = data_path + '\\' + train_lbl

    patcher = Patcher.from_image(train_img, train_lbl, _dim=(patch_size, patch_size), _stride=(128, 128))
    patches, labels = patcher.patchify()

    x_train_tmp = x_train_tmp + patches
    y_train_tmp = y_train_tmp + labels

    # if len(x_train_tmp) > 1:
    #     if first:
    #         x_train = np.array(x_train_tmp)
    #         y_train = np.array(y_train_tmp)
    #         first = False
    #     else:
    #         x_train = np.concatenate((x_train, np.array(x_train_tmp)))
    #         y_train = np.concatenate((y_train, np.array(y_train_tmp)))

    #     x_train_tmp = []
    #     y_train_tmp = []

    
    patcher = None
    gc.collect()

x_train = np.array(x_train_tmp)
y_train = np.array(y_train_tmp)

with h5py.File(savename, "w") as f:
    f.create_dataset('x_train', data=x_train)
    f.create_dataset('y_train', data=y_train)
