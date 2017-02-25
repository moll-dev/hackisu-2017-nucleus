""" Converts a depth image to a point cloud """

from PIL import Image
import numpy as np
#import calibkinect
#import pcl
#from pcl import registration

#import matplotlib.pyplot as plt

import ctypes

libpcgen = ctypes.cdll.LoadLibrary('./libpcgen.so')

def main():
        
    img = Image.open("test.png")
    #img = Image.open("images/capture745d.jpg")

    depth = np.array(img, dtype=np.float32)
    np_pc = depth_to_cloud(depth, 10)
    
    point_cloud  = pcl.PointCloud(np_pc)

    fil = point_cloud.make_statistical_outlier_filter()
    fil.set_mean_k (50)
    fil.set_std_dev_mul_thresh (1.0)
    
    pcl.save(point_cloud, "output0.ply")

def depth_to_cloud(depth, dist):

    if len(depth.shape) == 2:
        isPng = True
    else:
        isPng = False

    depth = depth / 255.0

    print(depth.shape)
    w = depth.shape[1]
    h = depth.shape[0]

    depth.flatten()
    num_pts = libpcgen.get_num_points(ctypes.c_void_p(depth.ctypes.data), ctypes.c_int(w*h))
    print(num_pts)
    out_pts = np.zeros((num_pts * 3), dtype=np.float32)
    libpcgen.build_pc(ctypes.c_void_p(depth.ctypes.data), ctypes.c_int(w*h), ctypes.c_void_p(out_pts.ctypes.data), ctypes.c_float(dist), ctypes.c_int(w), ctypes.c_int(h))
    
    return out_pts.reshape((num_pts, 3))

if __name__ == "__main__":
    main()