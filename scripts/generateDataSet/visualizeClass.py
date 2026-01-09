import open3d as o3d
import os
import numpy as np
import sys
sys.path.append(os.path.abspath('../..')) 


if __name__ == '__main__':
    pc = np.load('/home/rustam/ProjectMy/artifacts/dataCls/train/train_148.npy')
    print(pc.shape)
    obs_cloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc[1][..., 0:3]))
    
    # draw the axis
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([obs_cloud, coordinate])
    

   