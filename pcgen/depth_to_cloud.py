""" Converts a depth image to a point cloud """

from PIL import Image
import numpy as np
#import calibkinect
#import pcl
#from pcl import registration

import time
import ctypes
import os

if os.name == 'nt':
    print('NT')
    libpcgen = ctypes.cdll.LoadLibrary('.\\libpcgen.dll')
else:
    print('Linux')
    libpcgen = ctypes.cdll.LoadLibrary('./libpcgen.so')

def main():
        
    img = Image.open("test.png")
    d0, d1 = img.size[0], img.size[1]
    #img = img.resize((int(d0/8.0), int(d1/8.0)))
    #img = Image.open("images/capture745d.jpg")

    depth = np.array(img, dtype=np.float32)

    print('Cloudifying...')
    start = time.time()
    for i in range(100):
        np_pc = depth_to_cloud(depth, 10)
    end = time.time()
    t = (end - start) / 100 * 1000
    print('Time: {}ms'.format(t))

def depth_to_cloud(depth, dist):

    if len(depth.shape) == 2:
        isPng = True
    else:
        isPng = False

    depth = depth / 255.0

    h = depth.shape[0]
    w = depth.shape[1]
    
    zscale = 0.1
    libpcgen.set_bounds(ctypes.c_float(0.2), ctypes.c_float(0.8))

    depth.flatten()
    num_pts = libpcgen.get_num_points(ctypes.c_void_p(depth.ctypes.data), ctypes.c_int(w*h))
    out_pts = np.zeros((num_pts * 3), dtype=np.float32)
    libpcgen.build_pc(ctypes.c_void_p(depth.ctypes.data), ctypes.c_int(w*h), ctypes.c_void_p(out_pts.ctypes.data), ctypes.c_float(dist), ctypes.c_int(w), ctypes.c_int(h), ctypes.c_float(zscale))
    
    return out_pts.reshape((num_pts, 3))

if __name__ == "__main__":
    main()