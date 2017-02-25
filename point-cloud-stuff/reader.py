import pcl
import numpy as np
from pcl.registration import icp

def main():
    
    point_cloud_0 = pcl.load("data/bunny/bun0.pcd")
    point_cloud_1 = pcl.load("data/bunny/bun1.pcd")
    point_cloud_2 = pcl.load("data/bunny/bun2.pcd")
    point_cloud_3 = pcl.load("data/bunny/bun3.pcd")

    arr = np.asarray(point_cloud_1)
    #print(arr[:20])

    print("Loaded Point Clouds")
    #converged, transf, estimate, fitness = icp(point_cloud_1, point_cloud_2)

    estimate, converged = join_clouds((point_cloud_0, point_cloud_1, point_cloud_2, point_cloud_3))
    
    print("Converged? : {}".format(converged))
    
    pcl.save(estimate, "estimate.ply")


def join_clouds(clouds):
    
    if len(clouds) < 2:
        raise Exception("Not enough point clouds!")

    # Merge the first two clouds
    converged, transf, estimate, fitness = icp(clouds[0], clouds[1])
    print(converged)

    for cloud in clouds[2:]:
        converged, transf, estimate, fitness = icp(estimate, cloud)
        if not converged:
            print("DISCONVERGENCE")
    
    print(converged)
    return estimate, converged


if __name__ == "__main__":
    print("hello ^_^")
    main()