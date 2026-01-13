import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import numpy as np
from dexart.env.task_setting import ROBUSTNESS_INIT_CAMERA_CONFIG
import open3d as o3d
from examples.utils import get_viewpoint_camera_parameter, visualize_observation
from scripts.PointEncoderScripts.utils import SemSegDataset
import random


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


import numpy as np
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scripts.PointEncoderScripts.utils import SemSegDataset

import torch
from torch.utils.data import random_split
import torch.optim as optim

from torch.utils.data import  DataLoader
import os
import numpy as np
import argparse

from EncoderModels.PointNet import   PointNetSegBackbone, PointNetSeg
def visualize_segmented_pc(labels, points, use_seg=False, img_type=None):
    def visualize_pc_with_seg_label(cloud):
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud[:, :3]))

        def map(feature):
            color = np.zeros((feature.shape[0], 3))
            COLOR20 = np.array(
                [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                 [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190],
                 [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
                 [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128]]) / 255
            for i in range(feature.shape[0]):
                for j in range(feature.shape[1]):
                    if feature[i, j] == 1:
                        color[i, :] = COLOR20[j, :]
            return color

        color = map(cloud[:, 3:])
        pc.colors = o3d.utility.Vector3dVector(color)
        return pc

    pc = points
    if use_seg:
        gt_seg = labels
        pc = np.concatenate([pc, gt_seg], axis=1)
    pc = visualize_pc_with_seg_label(pc)
    return pc

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--extractor_name', type=str, default="smallpn")
    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--task', type=str, default=None)
    args = parser.parse_args()


    extractor_name = args.extractor_name
    pretrain_path = args.pretrain_path
    task =  args.task
  

  
    print('PointNetArchetecture')
    cat = 'bucket'


    backbone = PointNetSegBackbone()
    model = PointNetSeg(backbone, num_classes=4)
    model_state_dict = torch.load('/home/rustam/ProjectMy/artifacts/Encoders/PointNetSeg/fullModel.pth') 
    model.load_state_dict(model_state_dict)    
    
    dataset = SemSegDataset(root_dir='/home/rustam/ProjectMy/artifacts/DataSeg/bucket', split='train')
    print(len(dataset))
    # idx = np.random.randint(0, len(dataset))
    idx = 129
    pc, label = dataset[idx]
    points = pc.float().unsqueeze(0)
    outputs = model(points)               # [B, N, 4]
    outputs = outputs.permute(0, 2, 1)    # [B, 4, N]

    preds = outputs.argmax(dim=1).squeeze()
    colors = plt.get_cmap("tab20")(label / 4).reshape(-1, 4)
    obs_cloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc[..., 0:3]))
    obs_cloud.colors = o3d.utility.Vector3dVector(colors[:, 0:3])
    # draw the axis
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

    # o3d.visualization.draw_geometries([obs_cloud, coordinate])
    # colors = plt.get_cmap("tab20")(preds / 4).reshape(-1, 4)
    # obs_cloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc[..., 0:3]))
    # obs_cloud.colors = o3d.utility.Vector3dVector(colors[:, 0:3])
    # # draw the axis
    # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([obs_cloud, coordinate])
    
    

   