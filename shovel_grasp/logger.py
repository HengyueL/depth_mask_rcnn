import time
import datetime
import os
import numpy as np
import cv2
import torch 
# import h5py 


class Logger():
    def __init__(self, continue_logging, logging_directory, logger_dir=None):

        # Create directory to save data
        timestamp = time.time()
        timestamp_value = datetime.datetime.fromtimestamp(timestamp)
        self.continue_logging = continue_logging
        if self.continue_logging:
            self.base_directory = logging_directory
            print('Pre-loading data logging session: %s' % self.base_directory)
        elif logger_dir is None:
            self.base_directory = os.path.join(logging_directory, timestamp_value.strftime('%Y-%m-%d.%H:%M:%S'))
            print('Creating data logging session: %s' % self.base_directory)
        else:
            self.base_directory = os.path.join(logging_directory, logger_dir)
            print('Creating data logging session: %s' % self.base_directory)

        self.info_directory = os.path.join(self.base_directory, 'info')
        self.color_heightmaps_directory = os.path.join(self.base_directory, 'data', 'color-heightmaps')
        self.depth_heightmaps_directory = os.path.join(self.base_directory, 'data', 'depth-heightmaps')
        self.visualizations_directory = os.path.join(self.base_directory, 'visualizations')
        self.transitions_directory = os.path.join(self.base_directory, 'transitions')
        self.models_directory = os.path.join(self.transitions_directory, 'DQN_models')

        # -------- transition save dir ---------------------------
        self.action_dir = os.path.join(self.transitions_directory, 'actions')
        self.mask_dir = os.path.join(self.transitions_directory, 'mask')
        self.grasp_success_dir = os.path.join(self.transitions_directory, 'grasp-success')
        self.training_loss_dir = os.path.join(self.transitions_directory, 'training_loss')
        self.terminate_state_dir = os.path.join(self.transitions_directory, 'is-terminate')
        self.episode_reward_dir = os.path.join(self.transitions_directory, 'episode-reward-log')

        if not os.path.exists(self.info_directory):
            os.makedirs(self.info_directory)
        if not os.path.exists(self.color_heightmaps_directory):
            os.makedirs(self.color_heightmaps_directory)
        if not os.path.exists(self.depth_heightmaps_directory):
            os.makedirs(self.depth_heightmaps_directory)
        if not os.path.exists(self.models_directory):
            os.makedirs(self.models_directory)
        if not os.path.exists(self.visualizations_directory):
            os.makedirs(self.visualizations_directory)
        if not os.path.exists(self.transitions_directory):
            os.makedirs(os.path.join(self.transitions_directory, 'data'))
        # ------- Voxel method additional saver --------------------
        if not os.path.exists(self.mask_dir):
            os.makedirs(self.mask_dir)
        if not os.path.exists(self.training_loss_dir):
            os.makedirs(self.training_loss_dir)
        if not os.path.exists(self.action_dir):
            os.makedirs(self.action_dir)
        if not os.path.exists(self.grasp_success_dir):
            os.makedirs(self.grasp_success_dir)
        if not os.path.exists(self.terminate_state_dir):
            os.makedirs(self.terminate_state_dir)
        if not os.path.exists(self.episode_reward_dir):
            os.mkdir(self.episode_reward_dir)

    def save_camera_info(self, intrinsics, pose, depth_scale):
        np.savetxt(os.path.join(self.info_directory, 'camera-intrinsics.txt'), intrinsics, delimiter=' ')
        np.savetxt(os.path.join(self.info_directory, 'camera-pose.txt'), pose, delimiter=' ')
        np.savetxt(os.path.join(self.info_directory, 'camera-depth-scale.txt'), [depth_scale], delimiter=' ')

    def save_heightmap_info(self, boundaries, resolution):
        np.savetxt(os.path.join(self.info_directory, 'heightmap-boundaries.txt'), boundaries, delimiter=' ')
        np.savetxt(os.path.join(self.info_directory, 'heightmap-resolution.txt'), [resolution], delimiter=' ')

    def save_heightmaps(self, iteration, color_heightmap, depth_heightmap, mode):
        color_heightmap = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.color_heightmaps_directory, '%d.%s.color.png' % (iteration, mode)), color_heightmap)
        depth_heightmap = np.round(depth_heightmap * 100000).astype(np.uint16) # Save depth in 1e-5 meters
        cv2.imwrite(os.path.join(self.depth_heightmaps_directory, '%d.%s.depth.png' % (iteration, mode)), depth_heightmap)
    
    def write_to_log(self, log_name, log):
        np.savetxt(os.path.join(self.transitions_directory, '%s.log.txt' % log_name), log, delimiter=' ')

    def write_to_npy(self, log_name, array):
        np.save(os.path.join(self.transitions_directory, '%s.npy' % log_name), array)

    def save_model(self, iteration, model, name):
        torch.save(model.state_dict(), os.path.join(self.models_directory, '%d_%s.pth' % (iteration, name)))

    def save_visualizations(self, iteration, affordance_vis, name):
        cv2.imwrite(os.path.join(self.visualizations_directory, '%d.%s.png' % (iteration,name)), affordance_vis)

    def save_npy(self, array_to_save, iteration, directory):
        path = os.path.join(directory, '%d.npy' % iteration)
        np.save(path, array_to_save)
