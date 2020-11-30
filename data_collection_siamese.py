#!/usr/bin/env python
# This is the old script to collect object image patches to train Siamese Network
# Newer version should crop a image patch with a fixed size from the workspace (fully centered, nether too big or small)
# For better Siamese Detection

import time
import os
import argparse
import numpy as np
import cv2, torch
from shovel_grasp.Robot import Robot
import matplotlib.pyplot as plt
import shovel_grasp.utils as utils
from main_kidqn import get_heightmaps
from maskrcnn_training.sd_model import get_model_instance_segmentation
from skimage.color import rgb2gray


def transform_img(img, is_wall=False):
    """"
    transform 4 channel img into [gray, gray depth]
    img - ndarray of size (4, n, n), channel order (R, G, B, depth)
    """
    img_rgb = (255 * img[:, :, 0:3]).astype(np.uint8)
    img_gray = rgb2gray(img_rgb)
    img_depth = img[:, :, 3]
    # Normalize Height channel
    if is_wall:
        depth_max = 1.
    else:
        depth_max = 0.2

    img_depth = img_depth / depth_max
    x, y = img_depth.shape
    img_depth.shape = (x, y, 1)
    img_gray.shape = (x, y, 1)
    target_img = np.concatenate((img_gray, img_depth, img_depth),
                                axis=2)
    target_img = np.transpose(target_img, [2, 0, 1])
    return target_img


def get_rcnn_output(mask_rcnn_model,
                    color_heightmap,
                    depth_heightmap,
                    heightmap_size,
                    device):
    """
    :param mask_rcnn_model:  Pretrained Mask-RCNN model
    :param color_heightmap:  Workspace Color-heightmap
    :param depth_heightmap:  Workspace Depth-heightmap
    :param heightmap_size:
    :param device:           torch cpu or gpu
    :return:  two sets representing masks and bounding boxes
    """
    dept_data = depth_heightmap.copy()
    dept_data.shape = (heightmap_size,
                       heightmap_size,
                       1)
    rcnn_input = transform_img(np.concatenate((color_heightmap,
                                               dept_data), axis=2),
                               is_wall=False)
    rcnn_input = torch.from_numpy(rcnn_input).to(device, dtype=torch.float)
    rcnn_input.requires_grad = False
    rcnn_input = [rcnn_input]
    rcnn_output = mask_rcnn_model(rcnn_input)[0]
    mask_set = rcnn_output['masks'].cpu().data.detach().numpy()
    box_set = rcnn_output['boxes'].cpu().data.detach().numpy()
    # label_set = rcnn_output['labels'].cpu().data.detach().numpy()
    # score_set = rcnn_output['scores'].cpu().data.detach().numpy()
    return mask_set, box_set


def main(args):
    key_word = 'block'  # This key-word relates to the dir to save the collected samples
    # Initialize the V-rep env
    heightmap_size = 500
    obj_root_dir = args.obj_root_dir
    num_obj = args.num_obj
    save_root_dir = 'datasets/new_obj_set'
    workspace_limits = np.asarray([[-0.8, -0.3],
                                   [-0.2, 0.3],
                                   [0.0002,
                                    0.6002]])
    heightmap_resolution = float((workspace_limits[0][1] - workspace_limits[0][0])) / heightmap_size

    random_seed = args.random_seed
    np.random.seed(random_seed)
    # Initialize pick-and-place system (camera and robot)
    robot = Robot(obj_root_dir,
                  num_obj,
                  workspace_limits)
    # ===== Initialize Mask-RCNN model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    mask_rcnn_model = get_model_instance_segmentation(num_classes=2,
                                                      pretrained=False).to(device)
    mask_rcnn_model_path = args.mask_rcnn_model_path
    mask_rcnn_model.load_state_dict(torch.load(mask_rcnn_model_path))
    print('Mask RCNN model preload: ', mask_rcnn_model_path)
    mask_rcnn_model.eval()

    # save_model_dir = 'save_file_dir/Siamese_recollect/siamese-99.pth'
    # Create Dir to save the collected pathes
    save_obj_dir = os.path.join(save_root_dir, key_word)
    if not os.path.exists(save_obj_dir):
        os.mkdir(save_obj_dir)
    num_iter = 0
    while num_iter < 51:

        # Choose which object to add in the workspace accordingly
        obj_handle = robot.add_target()
        # obj_handle = robot.add_wall()

        # Capture heightmaps and object masks
        time.sleep(1)
        color_heightmap, depth_heightmap = get_heightmaps(robot,
                                                          workspace_limits,
                                                          heightmap_resolution=heightmap_resolution)
        mask_set, box_set = get_rcnn_output(mask_rcnn_model=mask_rcnn_model,
                                            color_heightmap=color_heightmap,
                                            depth_heightmap=depth_heightmap,
                                            heightmap_size=heightmap_size,
                                            device=device)
        if len(box_set) > 0:
            obj_center_x = int((box_set[0][0] + box_set[0][2]) / 2)
            obj_center_y = int((box_set[0][1] + box_set[0][3]) / 2)
            crop_x = obj_center_x - 75 if obj_center_x - 75 > 0 else 0
            crop_y = obj_center_y - 75 if obj_center_y - 75 > 0 else 0
            color_crop = color_heightmap[crop_y:crop_y+150,
                                         crop_x:crop_x+150,
                                         :]
            cv2.imwrite('color_crop.png', color_crop)
            depth_crop = depth_heightmap[crop_y:crop_y+150,
                                         crop_x:crop_x+150]
            x, y = depth_crop.shape
            depth_crop.shape = (x, y, 1)
            save_data = np.concatenate((color_crop, depth_crop),
                                       axis=2)
            np.save(os.path.join(save_obj_dir, '%d.npy' % num_iter), save_data)

            robot.stop_sim()
            robot.restart_sim()
            num_iter += 1
    print('Collection Done')


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--obj_root_dir', dest='obj_root_dir', action='store', default='shovel_grasp/objects',
                        help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--mask_rcnn_model_path', dest='mask_rcnn_model_path', action='store',
                        default='save_file_dir/pytorch_gdd_test/29.pth',
                        help='number of objects to add to simulation')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=1,
                        help='number of objects to add to simulation')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=17,
                        help='random seed for simulation and neural net initialization')
    # ------------- Algorithm options -------------
    parser.add_argument('--future_reward_discount', dest='future_reward_discount',
                        type=float, action='store', default=0.95)

    # ------ Pre-loading and logging options ------
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False,
                        help='continue logging from previous session?')
    parser.add_argument('--logging_directory', dest='logging_directory', action='store',
                        default='logs/2019-07-27.13:55:01')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=True,
                        help='save visualizations of FCN predictions?')
    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
