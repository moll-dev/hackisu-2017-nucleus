import matplotlib.pyplot as plt
import numpy as np
from patcher import Patcher
import os
import random
from sklearn.feature_extraction import image
import h5py
import gc
import sys

dataset = sys.argv[1]
savename = sys.argv[2]

data_path = 'C:\\nucleus\\'  + dataset
patch_size = 512

imgs = os.listdir(data_path)

x_train = None
x_train_tmp = []
y_train = None
y_train_tmp = []

first = True

# Patching data
for train_img in imgs:
    print(train_img)
    train_img = data_path + '\\Input\\' + train_img
    train_lbl = train_img.replace('Input\\', 'Label_Thresh\\').replace('JPG', 'png')

    patcher = Patcher.from_image(train_img, train_lbl, _dim=(patch_size, patch_size), _stride=(128, 128))
    patches, labels = patcher.patchify(random=True, max_patches=25)

    x_train_tmp = x_train_tmp + patches
    y_train_tmp = y_train_tmp + labels

    if len(x_train_tmp) > 2000:
        if first:
            x_train = np.array(x_train_tmp)
            y_train = np.array(y_train_tmp)
            first = False
        else:
            x_train = np.concatenate((x_train, np.array(x_train_tmp)))
            y_train = np.concatenate((y_train, np.array(y_train_tmp)))

        x_train_tmp = []
        y_train_tmp = []

    
    patcher = None
    gc.collect()

x_train = np.array(x_train)
y_train = np.array(y_train)


with h5py.File('C:\\nucleus\\' + savename, "w") as f:
    f.create_dataset('images', data=x_train)
    f.create_dataset('labels', data=y_train)

print(test_imgs)