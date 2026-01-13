import open3d as o3d
import os
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import matplotlib.pyplot as plt
from scripts.PointEncoderScripts.utils import ClsDataset


if __name__ == '__main__':
    dataset = ClsDataset('/home/rustam/ProjectMy/artifacts/dataCls/train.npz')
    print(len(dataset))
    idx = np.random.randint(0, len(dataset))
    pc, label = dataset[idx]

    obs_cloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc[..., 0:3]))
    # draw the axis
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([obs_cloud, coordinate])
    

   