import numpy as np
import os.path
from glob import glob
from dexart.env.task_setting import TRAIN_CONFIG, RANDOM_CONFIG
from dexart.env.create_env import create_env
from tqdm import tqdm
import argparse

def gen_single_data(task_name, index, label, split,
                    n_fold=32, img_type='robot',
                    save_path='/home/rustam/ProjectMy/artifacts/dataCls'):

    env = create_env(
        task_name=task_name,
        use_visual_obs=True,
        use_gui=False,
        is_eval=False,
        pc_noise=True,
        pc_seg=True,
        index=[index],
        img_type=img_type,
        rand_pos=RANDOM_CONFIG[task_name]['rand_pos'],
        rand_degree=RANDOM_CONFIG[task_name]['rand_degree'],
    )

    obs = env.reset()
    data = []

    for i in tqdm(range(env.horizon * n_fold)):
        action = np.random.uniform(-2, 2, size=env.action_space.shape)
        obs, reward, done, _ = env.step(action)

        qlimits = env.instance.get_qlimits()
        qpos = np.random.uniform(qlimits[:, 0], qlimits[:, 1])
        env.instance.set_qpos(qpos)

        observed_pc = np.concatenate(
            [obs['instance_1-point_cloud'], obs['imagination_robot'][:, :3]],
            axis=0
        )
        assert observed_pc.shape == (608, 3)

        data.append(observed_pc)
        env.scene.update_render()

        if i % env.horizon == 0:
            obs = env.reset()

    data = np.array(data, dtype=np.float32)          # [B, 608, 3]
    labels = np.full((len(data),), label, np.int64)  # [B]

    split_dir = os.path.join(save_path, split)
    os.makedirs(split_dir, exist_ok=True)

    np.savez(
        os.path.join(split_dir, f"{split}_{index}.npz"),
        points=data,
        labels=labels
    )

    print(f"save {split}_{index}.npz")



def merge_data(save_path='/home/rustam/ProjectMy/artifacts/dataCls'):

    for split in ['train', 'val', 'test']:
        all_points = []
        all_labels = []

        files = glob(os.path.join(save_path, split, f"{split}_*.npz"))

        for f in files:
            data = np.load(f)
            all_points.append(data['points'])
            all_labels.append(data['labels'])
            print(f)

        points = np.concatenate(all_points, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        np.savez(
            os.path.join(save_path, f"{split}.npz"),
            points=points,
            labels=labels
        )

        print(f"{split}: {points.shape}, {labels.shape}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='faucet')
    args = parser.parse_args()

    save_path='/home/rustam/ProjectMy/artifacts/dataCls'
    mapping = {'faucet': 1, "bucket":2, "laptop":3, "toilet":4}
    tasks = ['faucet', 'bucket', 'laptop', 'toilet']
    for task_name in tasks:
        for index in TRAIN_CONFIG[task_name]['seen']:
            gen_single_data(task_name, index, mapping[task_name], 'train', n_fold=2, save_path=save_path)
        for index in TRAIN_CONFIG[task_name]['seen']:
            gen_single_data(task_name, index, mapping[task_name], 'val', n_fold=1, save_path=save_path)
        for index in TRAIN_CONFIG[task_name]['unseen']:
            gen_single_data(task_name, index, mapping[task_name], 'test', n_fold=1, save_path=save_path)
    merge_data(save_path=save_path)
