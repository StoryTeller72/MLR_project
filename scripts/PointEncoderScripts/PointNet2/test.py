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
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

train_dataset = SemSegDataset(split='train', point_channel=3, use_img=False, root_dir=f'/home/rustam/ProjectMy/artifacts/DataSeg/bucket')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

data_iterator = iter(train_loader)
points, labels = next(data_iterator)

points = points.float()
labels = labels.long()
model = PointNet2.PointNet2Lite(4)
res = model(points)
print('final_shape',res.shape)