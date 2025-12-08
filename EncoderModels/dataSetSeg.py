import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import numpy as np

class PointCloudDatasetSet(Dataset):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.file_pairs = []
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
            segment_labels = np.load(seg_path).astype(np.int64) 
        except Exception as e:
            print(f"Ошибка загрузки файлов {pc_path} или {seg_path}: {e}")
            return self[idx + 1] 

        point_cloud = torch.from_numpy(point_cloud)
        segment_labels = torch.from_numpy(segment_labels)
        return point_cloud, segment_labels

