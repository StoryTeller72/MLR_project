from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from os.path import join as pjoin
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scripts.PointEncoderScripts.utils import SemSegDataset


# test
if __name__ == '__main__':
    dataset = SemSegDataset(root_dir='/home/rustam/ProjectMy/artifacts/DataSeg/bucket', split='train')
    print(len(dataset))
    idx = np.random.randint(0, len(dataset))
    pc, label = dataset[idx]
    print(label.shape)
    colors = plt.get_cmap("tab20")(label / 4).reshape(-1, 4)
    obs_cloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc[..., 0:3]))
    obs_cloud.colors = o3d.utility.Vector3dVector(colors[:, 0:3])
    # draw the axis
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([obs_cloud, coordinate])