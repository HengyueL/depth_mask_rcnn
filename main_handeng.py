# This script is the hand engineered policy baselines
import numpy as np
import os
import argparse
# from shovel_grasp.Robot import Robot
from shovel_grasp.Robot_test import Robot
from shovel_grasp.logger import Logger
import matplotlib.pyplot as plt
import time
from main_kidqn import get_heightmaps
from scipy import ndimage


def main(args,
         logger_dir,
         method=None):
    assert method is not None, 'Method cannot be NONE. Check input param please.'
    # A few things to log
    num_of_episode = 0
    num_of_stuck = 0
    test_action_count = 0
    episode_action_count = 0
    episode_reward_log = []

    # --------------- Setup options ---------------
    obj_root_dir = os.path.abspath(args.obj_root_dir)
    num_obj = args.num_obj
    random_seed = args.random_seed
    # Set random seed
    np.random.seed(random_seed)
    rotation_range = 90.
    rotation_interval = 5
    max_interval_idx = int(rotation_range/rotation_interval)
    heightmap_size = 500
    # Modify the following action space size according to the model output size
    action_space_size = 64
    workspace_limits = np.asarray([[-0.8, -0.3],
                                   [-0.2, 0.3],
                                   [0.0002,
                                    0.6002]])
    # Image and voxel grid parameters
    heightmap_resolution = float(workspace_limits[0][1] - workspace_limits[0][0]) / heightmap_size
    print('heightmap_resolution: ', heightmap_resolution)
    action_space_resolution = float(workspace_limits[0][1] - workspace_limits[0][0]) / action_space_size
    # ------ Pre-loading and logging options ------
    continue_logging = False
    logging_directory = os.path.abspath('logs')
    save_visualizations = args.save_visualizations
    # Initialize pick-and-place system (camera and robot)
    robot = Robot(obj_root_dir,
                  num_obj,
                  workspace_limits)
    # Initialize data logger
    logger_dir = os.path.join(logger_dir,
                              method)
    logger = Logger(continue_logging,
                    logging_directory,
                    logger_dir=logger_dir)

    no_change_count = 0
    while True:
        iteration_time_0 = time.time()
        sim_ok = robot.check_sim()
        if not sim_ok:
            robot.restart_sim()
            episode_action_count = 0
        time.sleep(0.5)
        restart_flag = False
        color_heightmap, depth_heightmap = get_heightmaps(robot,
                                                          workspace_limits,
                                                          heightmap_resolution)
        if method != 'human_input':
            obj_pos = robot.get_single_obj_position(robot.target_handles[0])
            # wall_pos = robot.get_single_obj_position(robot.wall_handles[0])
            # angle_wall_obj = np.arctan2(wall_pos[1]-obj_pos[1],
            #                             wall_pos[0]-obj_pos[0])

        if method == 'human':
            print('Human Method?   ', method)
            # deg = np.rad2deg(angle_wall_obj) - 90
            # interval_idx = min(int(deg / rotation_interval),
            #                    max_interval_idx)
            pass
        elif method == 'random':
            print('Random Method?   ', method)
            interval_idx = np.random.randint(0, max_interval_idx+1)
            random_pos = np.random.randint(-9, 10, size=(2,))
            pos_0 = 0.5 / 64 * random_pos[0]
            pos_1 = 0.5 / 64 * random_pos[1]
            obj_pos[0] = obj_pos[0] + pos_0
            obj_pos[1] = obj_pos[1] + pos_1
        elif method == 'human_input':
            zoom_scale = action_space_size / heightmap_size
            vis_map = ndimage.zoom(depth_heightmap,
                                   zoom=[zoom_scale, zoom_scale],
                                   mode='constant',
                                   prefilter=False)
            plt.imshow(vis_map)
            plt.show()
            angle = input('Input Shovel Angle: (0 - 90)')
            interval_idx = int(angle) // rotation_interval
            x = input('Input Shovel Angle x point')
            pos_x = (int(x)+ 0.5) / zoom_scale * heightmap_resolution + workspace_limits[0][0]
            y = input('Input Shovel Angle y point')
            pos_y = (int(y) +0.5)/ zoom_scale * heightmap_resolution + workspace_limits[1][0]
            obj_pos = np.asarray([pos_x, pos_y, 0.0])
        else:
            print('Incorrect method specification. Check input param!')
            break

        if interval_idx < 0:
            shovel_success = robot.shovel(obj_pos,
                                          0)
        else:
            shovel_success = robot.shovel(obj_pos,
                                          interval_idx * rotation_interval)
        episode_action_count = episode_action_count + 1
        print('Shovel Successul: ', shovel_success)
        if shovel_success:
            restart_flag = True
            episode_reward_log.append(1)
        else:
            no_change_count += 1
        is_stuck = no_change_count > 10
        if is_stuck:
            num_of_stuck += 1
            restart_flag = True
        if len(robot.target_handles) > 0:
            obj_pos = robot.get_single_obj_position(robot.target_handles[0])
            if obj_pos[0] < workspace_limits[0][0] + 0.02 \
                    or obj_pos[0] > workspace_limits[0][1] - 0.02 \
                    or obj_pos[1] < workspace_limits[1][0] + 0.02 \
                    or obj_pos[1] > workspace_limits[1][1] - 0.02:
                restart_flag = True

        if restart_flag:
            # if shovel_success is True:
            #     episode_reward_log.append(1)
            # else:
            #     episode_reward_log.append(0)
            num_of_episode = num_of_episode + 1
            test_action_count += episode_action_count
            episode_action_count = 0
            no_change_count = 0
            print('Restart Simulation.')
            robot.stop_sim()
            robot.restart_sim()
            restart_flag = False
        logger.save_npy(np.asarray(episode_reward_log),
                        0,
                        logger.episode_reward_dir)

        logger.save_npy(np.asarray(num_of_stuck),
                        0,
                        logger.transitions_directory)
        logger.save_npy(np.asarray(num_of_episode),
                        1,
                        logger.transitions_directory)
        logger.save_npy(np.asarray(test_action_count),
                        2,
                        logger.transitions_directory)

        if num_of_episode >= 15:
            robot.stop_sim()
            break
        print('Number Success: ', np.sum(episode_reward_log))
        print('Episode Num: %d' % num_of_episode)
        print('Stuck Number: ', num_of_stuck)
        print('Episode action count: ', episode_action_count)
        print('Total action count: ', test_action_count)
        print()


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')
    # --------------- Setup options ---------------
    parser.add_argument('--obj_root_dir', dest='obj_root_dir', action='store', default='shovel_grasp/objects',
                        help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=1,
                        help='number of objects to add to simulation')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=17,
                        help='random seed for simulation and neural net initialization')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=True,
                        help='save visualizations of FCN predictions?')
    # Run main program with specified arguments
    args = parser.parse_args()
    num_rotations = 7
    method_list = ['random']  # options are: 'human_input', 'random'
    for method in method_list:
        print('Testing Method: ', method)
        main(args,
             logger_dir='test/baselines',
             method=method)

