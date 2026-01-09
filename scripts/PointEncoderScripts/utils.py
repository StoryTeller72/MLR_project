from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from os.path import join as pjoin
import torch

import os
import glob
import numpy as np


# train set: data/faucet_img/train.npy
class SemSegDataset(Dataset):
    def __init__(self, root_dir='../data/faucet', split='train', use_img=True, point_channel=3):
        self.root_dir = root_dir
        self.split = split
        self.data = self.load_data()
        self.use_img = use_img
        self.point_channel = point_channel
        self.labelweights = np.ones(4)

    def load_data(self):
        if self.split == 'train':
            data = np.load(pjoin(self.root_dir, 'train.npy'), 'r')
        elif self.split == 'val':
            data = np.load(pjoin(self.root_dir, 'val.npy'), 'r')
        elif self.split == 'test':
            data = np.load(pjoin(self.root_dir, 'test.npy'), 'r')
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.use_img:
            points = sample[:, 0:self.point_channel]
            labels = np.argmax(sample[:, 3:], axis=1)
        else:
            points = sample[0:512, 0:self.point_channel] 
            labels = np.argmax(sample[0:512, 3:], axis=1)
        return torch.tensor(points), torch.tensor(labels)
    
class ClsDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.points = data['points'].astype(np.float32)  # [B, N, 3]
        self.labels = data['labels'].astype(np.int64)    # [B]

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        points = self.points[idx]   # [N, 3]
        label = self.labels[idx]    # int

        points = torch.from_numpy(points)      # [N, 3]
        label = torch.tensor(label).long()     # []

        return points, label



