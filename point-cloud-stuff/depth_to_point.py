""" Converts a depth image to a point cloud """

from PIL import Image
import numpy as np
import calibkinect
import pcl
#from pcl import registration

def main():
        
    img = Image.open("images/capture736d.jpg")

    depth = np.array(img)
    np_pc = depth_to_cloud(depth, 1.0)
    
    point_cloud  = pcl.PointCloud(np_pc)

    fil = point_cloud.make_statistical_outlier_filter()
    fil.set_mean_k (50)
    fil.set_std_dev_mul_thresh (1.0)
    
    pcl.save(fil.filter(), "output.ply")
    #array = get_point_cloud(depth)
    #img.show()
    #point_cloud = pcl.PointCloud(array, dtype=np.float32)

def depth_to_cloud(depth, dist):
    w = depth.shape[1]
    h = depth.shape[0]
    
    dist_ray = np.array([0,0,dist])
    output = []
    float_w = float(w)

    for u in range(h):
        for v in range(w):
            u2 = u - w/2
            v2 = v - h/2
            x_p = u2 / float_w
            y_p = v2 / float_w

            d = depth[u][v][0] / 255.0
            xy_ray = np.array([x_p, y_p, 0])
            r = (xy_ray - dist_ray) / np.linalg.norm(xy_ray - dist_ray)
            ray = xy_ray + (d) * r

            if .2 < d < .7:
                output.append(ray)
    
    return np.array(output, dtype=np.float32)

if __name__ == "__main__":
    main()