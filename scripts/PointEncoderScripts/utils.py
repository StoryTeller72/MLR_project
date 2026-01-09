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
    
class PointCloudDatasetClf(Dataset):
    def __init__(self, base_dir, class_amnt, class_mapping):
        self.class_amnt = class_amnt
        self.base_dir = base_dir
        self.file_pairs = []
        self.mapping = class_mapping
        for category_name in os.listdir(base_dir):
            category_path = os.path.join(base_dir, category_name)
            if os.path.isdir(category_path):
                pc_dir = os.path.join(category_path, 'pc')
                seg_dir = os.path.join(category_path, 'seg')

                pc_files = sorted(glob.glob(os.path.join(pc_dir, '*.npy')))
                seg_files = sorted(glob.glob(os.path.join(seg_dir, '*.npy')))
                if len(pc_files) == len(seg_files) and len(pc_files) > 0:
                    for pc_path, seg_path in zip(pc_files, seg_files):
                        self.file_pairs.append({'pc': pc_path, 'seg': seg_path, 'class':category_name})
                else:
                    print(f"Внимание: Несоответствие файлов в категории {category_name}")

        if not self.file_pairs:
            raise RuntimeError(f"Не найдено ни одной пары файлов в директории {base_dir}")
            
        print(f"Всего найдено пар облаков точек: {len(self.file_pairs)}")

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
    
        pc_path = self.file_pairs[idx]['pc']
        seg_path = self.file_pairs[idx]['seg']
        object_class = self.file_pairs[idx]['class']
        try:
            point_cloud = np.load(pc_path).astype(np.float32)
        except Exception as e:
            print(f"Ошибка загрузки файлов {pc_path} или {seg_path}: {e}")
            return self[idx + 1] 

        point_cloud = torch.from_numpy(point_cloud)
        object_class= self.mapping[object_class]
        return (point_cloud, object_class)



