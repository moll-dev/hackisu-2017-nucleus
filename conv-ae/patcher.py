import numpy as np
from PIL import Image
import h5py
import random as rng
import matplotlib.pyplot as plt
from PIL import ExifTags
import scipy.misc

class Patcher():
    def __init__(self, _img_arr, _lbl_arr, _dim, _stride=(4,4), _patches=None, _labels=None):
        self.img_arr = _img_arr

        if _lbl_arr == None:
            _lbl_arr = np.ones((_img_arr.shape[0], _img_arr.shape[1]))
            
        self.lbl_arr = _lbl_arr
        self.dim = _dim
        self.stride = _stride
        self.patches = _patches
        self.labels = _labels

    @classmethod
    def from_image(cls, _img_file, _lbl_file, _dim=(32,32), _stride=(4,4)):
        img = Image.open(_img_file)
        d0, d1 = img.size[0], img.size[1]
        img = img.resize((int(d0/2.0), int(d1/2.0)))
        #img = img.resize((754, 424), Image.ANTIALIAS)
        #img = img.crop((121, 0, 633, 424))

        img_arr = np.array(img, dtype=np.float32)/255.0

        if _lbl_file == None:
            lbl_arr = None
        else:
            lbl = Image.open(_lbl_file)
            lbl_arr = np.array(lbl, dtype=np.float32)[:,:,0]/255.0

            assert img_arr.shape[0] == lbl_arr.shape[0]
            assert img_arr.shape[1] == lbl_arr.shape[1]

        return cls(img_arr, lbl_arr, _dim, _stride)

    def set_patch_dim(self, _dim):
        self.dim = _dim

    def create_patch(self, pos, flatten=False, label=False):
        d0 = self.dim[0]
        d1 = self.dim[1]

        shape = self.img_arr.shape

        d00 = pos[0]
        d01 = pos[0] + d0

        if d01 > shape[0]:
            d00 = d00 - (d01 - shape[0])
            d01 = d01 - (d01 - shape[0])

        d10 = pos[1]
        d11 = pos[1] + d1

        if d11 > shape[1]:
            d10 = d10 - (d11 - shape[1])
            d11 = d11 - (d11 - shape[1])

        if label:
            patch = self.lbl_arr[d00:d01, d10:d11]
        else:
            patch = self.img_arr[d00:d01, d10:d11]

        assert patch.shape[0] == d0
        assert patch.shape[1] == d1

        if flatten:
            return patch.flatten()
        else:
            if label:
                return patch.reshape((d0, d1, 1))
            else:
                return patch

    def patchify(self):
        if self.patches != None:
            return self.patches, self.labels

        self.patches = []
        self.labels = []

        shape = self.img_arr.shape

        d0 = self.dim[0]
        d1 = self.dim[1]
        s0 = self.stride[0]
        s1 = self.stride[1]

        for i0 in range(0, shape[0] - d0, s0): 
            for i1 in range(0, shape[1] - d1, s1):
                label_patch = self.create_patch([i0, i1], label=True)
                if np.sum(label_patch.flatten()) > 0  or rng.randint(0,100) < 25:
                    self.patches.append(self.create_patch([i0, i1], label=False))
                    self.labels.append(label_patch)

        return self.patches, self.labels

    def num_patches(self):
        return self.patches.shape[0]

    def predict(self, predictor, frac=1.0):
        pred_label = np.zeros_like(self.lbl_arr)
        shape = pred_label.shape

        d0 = self.dim[0]
        d1 = self.dim[1]

        d0_stride = int(d0 * frac)
        d1_stride = int(d1 * frac)

        patches = []

        # TODO:
        # This cuts off any part of the image not aligned with d0, d1, boundarys.
        # For small enough patch dimensions this isn't a huge deal, but still would
        # be a good idea to create a smarter algorithm here.

        for i0 in range(0, shape[0], d0_stride): 
            for i1 in range(0, shape[1], d1_stride):
                patches.append(self.create_patch([i0, i1], label=False))

        patches = np.array(patches)
        preds = predictor(patches)

        i = 0
        for i0 in range(0, shape[0], d0_stride): 
            for i1 in range(0, shape[1], d1_stride):
                if i0 + d0 > shape[0]:
                    if i1 + d1 > shape[1]:
                        pred_label[i0:, i1:] += preds[i].reshape((d0, d1))[d0 - (shape[0] - i0):, d1 - (shape[1] - i1):]
                    else:
                        pred_label[i0:, i1:i1+d1] += preds[i].reshape((d0, d1))[d0 - (shape[0] - i0):, :]
                elif i1 + d1 > shape[1]:
                    pred_label[i0:i0+d0, i1:] += preds[i].reshape((d0, d1))[:, d1 - (shape[1] - i1):]
                else:
                    pred_label[i0:i0+d0, i1:i1+d1] += preds[i].reshape((d0, d1))
                    
                i = i + 1

        #pred_label = np.where(pred_label > 0.7, 1, 0)

        return pred_label


