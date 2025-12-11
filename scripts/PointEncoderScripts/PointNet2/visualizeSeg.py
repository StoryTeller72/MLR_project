import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from EncoderModels import PointNet2
import torch
from torch.utils.data import random_split
import torch.optim as optim

from torch.utils.data import  DataLoader
import os
import numpy as np
import argparse
import sys
sys.path.append(os.path.abspath('')) 
from scripts.PointEncoderScripts.utils import SemSegDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
import open3d as o3d
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--extractor_name', type=str, default="pn2")
    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--task', type=str, default=None)
    args = parser.parse_args()


    extractor_name = args.extractor_name
    pretrain_path = args.pretrain_path
    task =  args.task
    
    model = PointNet2.PointNet2Lite(4)



    model.load_state_dict(torch.load("/home/rustam/ProjectMy/artifacts/PointNet2/1.pth"))

 
    train_dataset = SemSegDataset(split='train', point_channel=3, use_img=False, root_dir=f'/home/rustam/ProjectMy/artifacts/DataSeg/bucket')
    idx = np.random.randint(0, len(train_dataset))
    pc, label = train_dataset[idx]
    points = pc.float()
    points = torch.unsqueeze(points, 0)
    pred = model(points)
    pred = torch.softmax(pred, dim=-1)
    pred = torch.argmax(pred, dim=-1)
    colors = plt.get_cmap("tab20")(pred / 4).reshape(-1, 4)
    obs_cloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc[..., 0:3]))
    obs_cloud.colors = o3d.utility.Vector3dVector(colors[:, 0:3])
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([obs_cloud, coordinate])

    

   