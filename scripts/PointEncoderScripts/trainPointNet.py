from stable_baselines3.networks.pretrain_nets import PointNet, PointNetMedium, PointNetLarge
import torch
from dataSet import PointCloudDataset

import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import numpy as np




if __name__ == '__main__':
    dataSet = PointCloudDataset("../../dexartEnv/assets/data")