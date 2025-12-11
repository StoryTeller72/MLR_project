import numpy as np
import sys, os
from stable_baselines3.networks import pretrain_nets
# from scripts.PointEncoderScripts.utils import SemSegDataset
model = pretrain_nets.PointNet2Lite()
# train_dataset = SemSegDataset(split='train', point_channel=3, use_img=False, root_dir=f'/home/rustam/ProjectMy/artifacts/DataSeg/bucket')
