from sapien.utils import Viewer
from examples.utils import get_viewpoint_camera_parameter, visualize_observation
import argparse
import numpy as np
from dexart.env.create_env import create_env
import open3d as o3d

if __name__ == "__main__":
    path = 'dexartEnv/assets/data/segmentation'
    num_episodes = 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, required=True)
    args = parser.parse_args()
    task_name = args.task_name

    env = create_env(task_name=task_name, use_visual_obs=True, img_type='robot', use_gui=True, pc_seg=True)
    robot_dof = env.robot.dof
    
    mapping = {}

    # config the viewer
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    viewer.focus_camera(env.cameras['instance_1'])
    env.viewer = viewer
   
    origin, target, up, m44 = get_viewpoint_camera_parameter()
    iter = 0
    needData = 1000
    while iter < needData:
        obs = env.reset()
        for i in range(21):
            action = np.random.rand(22)
            obs, reward, done, info = env.step(action)
            viewer.render()
            pc = visualize_observation(obs, use_seg=True, img_type='robot')
            coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
            if i % 5 == 0:
                np.savetxt(f"/home/rustam/ProjectMy/dexartEnv/assets/data/segmentation/{task_name}/{task_name}_{iter}.txt", obs['instance_1-seg_gt'])
                np.savetxt(f"/home/rustam/ProjectMy/dexartEnv/assets/data/pointClouds/{task_name}/{task_name}_pc_{iter}.txt", obs['instance_1-point_cloud'])
                iter += 1
    o3d.visualization.draw_geometries([pc, coordinate], zoom=1,
                                      front=origin - target,
                                      lookat=target,
                                      up=up)
    viewer.close()