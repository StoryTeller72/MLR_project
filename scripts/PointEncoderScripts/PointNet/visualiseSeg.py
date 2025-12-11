import numpy as np
from dexart.env.task_setting import ROBUSTNESS_INIT_CAMERA_CONFIG
import open3d as o3d
from examples.utils import get_viewpoint_camera_parameter, visualize_observation

import random

import torch
from torch.utils.data import random_split
import torch.optim as optim

from torch.utils.data import  DataLoader
import os
import numpy as np
import argparse

import sys
sys.path.append(os.path.abspath('../..')) 
from EncoderModels.PointNet import PointNet, PointNetMedium, PointNetLarge, PointNetClassifier
from EncoderModels.PointNet import  PointNetSegmentationHead
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
    if extractor_name == "smallpn":
        net = PointNet()
    elif extractor_name == "mediumpn":
        net =  PointNetMedium()
    elif extractor_name == "largepn":
        net = PointNetLarge()

  
    print('PointNetArchetecture')
    print(net)

    print("PointNet with classification head")
    pointNetCls = PointNetClassifier(net, 4)
    if pretrain_path:
        pointNetCls.load_state_dict(torch.load(pretrain_path))

    print(pointNetCls)
    index = random.randint(0, 2_000)
    row_pc = np.load(f'../../dexartEnv/assets/data/{task}/pc/{index}.npy', allow_pickle = True)
    seg = np.load(f'../../dexartEnv/assets/data/{task}/seg/{index}.npy',allow_pickle=True)

    mapping = {0:"bucket", 1:'faucet', 2:"laptop", 3: "toilet"}
    output =pointNetCls(torch.from_numpy(row_pc).float().unsqueeze(0)) 
    label = mapping[int(output.argmax(1))]
    print(label)
    origin, target, up, m44 = get_viewpoint_camera_parameter()
    pc = visualize_segmented_pc(labels=seg, points=row_pc, use_seg=True)
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pc, coordinate], zoom=1,
                                      front=origin - target,
                                      lookat=target,
                                      up=up)
   
    

   