#!/usr/bin/env python
# Main Script of KI-DQN project (Training and Testing)

import time
import os
import argparse
import numpy as np
import cv2
from scipy import ndimage
from shovel_grasp.trainer import TrainerDQN
from shovel_grasp.trainer import get_prediction_vis
from shovel_grasp.logger import Logger
import shovel_grasp.utils as utils
import matplotlib.pyplot as plt
from maskrcnn_training.sd_model import get_model_instance_segmentation
import siamese_model
import torch
from skimage.color import rgb2gray


def get_distance(a, b):
    return np.sqrt(np.sum(np.power((a-b), 2)))


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


def process_input_imgs(color_heightmap, depth_heightmap, heightmap_size):
    """
    Pre-process the input heightmaps in order to be ready
    for the visual affordance network
    """
    pad_width = int((np.sqrt(2) - 1) * heightmap_size / 2) + 1
    DQN_input_data = np.zeros([pad_width * 2 + heightmap_size,
                               pad_width * 2 + heightmap_size,
                               4], dtype=np.float)

    DQN_color_input = color_heightmap.astype(float) / 255.
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    for c in range(3):
        DQN_input_data[pad_width:pad_width+heightmap_size,
                       pad_width:pad_width+heightmap_size,
                       c] = (DQN_color_input[:, :, c] - img_mean[c]) / img_std[c]

    height_scale = 0.2
    DQN_input_data[pad_width:pad_width + heightmap_size,
                   pad_width:pad_width + heightmap_size,
                   3] = depth_heightmap / height_scale
    return DQN_input_data


def compute_siamese_vector(siamese_model,
                           color_heightmap,
                           depth_heightmap,
                           mask,
                           heightmap_size,
                           device):
    """
    Compute the target object encoded vector by the Siamese network.
    :param siamese_model:
    :param color_heightmap: np.array (H, W, C)
    :param depth_heightmap: np.array (H, W)
    :param mask: np.array (H, W)
    :return:
    """
    color_input = np.zeros_like(color_heightmap)
    for c in range(color_heightmap.shape[2]):
        color_input[:, :, c] = np.multiply(color_heightmap[:, :, c], mask)
    depth_input = np.multiply(depth_heightmap,
                              mask)
    depth_input.shape = (heightmap_size, heightmap_size, 1)
    input_data = np.concatenate((color_input, depth_input),
                                axis=2)
    input_data = np.asarray([transform_img(input_data, is_wall=False)])
    input_data = torch.from_numpy(input_data).to(device=device, dtype=torch.float)
    input_data.requires_grad = False
    output_data = siamese_model(input_data).data.cpu().detach().numpy()[0]
    return output_data


def experience_replay(sample_iteration, logger, trainer,
                      output_size, heightmap_size, weight_scale=1e-1):
    """
    Wrapper for experience Replay process.
    Variables are self-explainable by their names.
    """
    sample_color_img = np.load(os.path.join(logger.color_heightmaps_directory,
                                            '%d.npy' % sample_iteration))
    sample_depth_img = np.load(os.path.join(logger.depth_heightmaps_directory,
                                            '%d.npy' % sample_iteration))
    sample_input_data = process_input_imgs(sample_color_img,
                                           sample_depth_img,
                                           heightmap_size)

    sample_action_position = np.load(os.path.join(logger.action_dir,
                                                  '%d.npy' % sample_iteration))
    sample_grasp_success = np.load(os.path.join(logger.grasp_success_dir,
                                                '%d.npy' % sample_iteration))
    sample_mask = np.load(os.path.join(logger.mask_dir,
                                       '%d.npy' % sample_iteration))
    sample_next_color_img = np.load(os.path.join(logger.color_heightmaps_directory,
                                                 '%d.npy' % (sample_iteration + 1)))
    sample_next_depth_img = np.load(os.path.join(logger.depth_heightmaps_directory,
                                                 '%d.npy' % (sample_iteration + 1)))
    sample_next_state_data = process_input_imgs(sample_next_color_img,
                                                sample_next_depth_img,
                                                heightmap_size)
    sample_terminate_state = np.load(os.path.join(logger.terminate_state_dir,
                                                  '%d.npy' % sample_iteration))

    sample_grasp_score, _ = trainer.get_label_value(sample_grasp_success,
                                                    sample_next_state_data,
                                                    sample_terminate_state)
    loss_value = trainer.backprop_mask(input_img=sample_input_data,
                                       rot_idx=sample_action_position[0],
                                       action_position=sample_action_position[1:3],
                                       output_size=output_size,
                                       label_value=sample_grasp_score,
                                       prev_mask=sample_mask,
                                       weight_scale=weight_scale)
    print('Replay Training Loss: %f' % loss_value)
    return loss_value


def get_heightmaps(robot, workspace_limits, heightmap_resolution):
    """
    Wrapper function to observe the heightmaps of the workspace
    """
    color_img, depth_img = robot.get_camera_data()
    depth_img = depth_img * robot.cam_depth_scale  # Apply depth scale from calibration
    color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics,
                                                           robot.cam_pose, workspace_limits, heightmap_resolution)
    valid_depth_heightmap = depth_heightmap.copy()
    valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

    kernel = np.ones([3, 3])
    color_heightmap = cv2.dilate(color_heightmap, kernel, iterations=2)
    color_heightmap = cv2.erode(color_heightmap, kernel, iterations=2)
    valid_depth_heightmap = cv2.dilate(depth_heightmap, kernel, iterations=2)
    valid_depth_heightmap = cv2.erode(valid_depth_heightmap, kernel, iterations=2)
    return color_heightmap, valid_depth_heightmap


def main(args,
         logger_dir,
         method=None,
         model_logger_dir=None,
         num_rotations=7,
         is_testing=False,
         filtered_exploration=False,
         mask_detection=False,
         continue_iteration=0):
    if not is_testing:
        from shovel_grasp.Robot import Robot
    else:
        from shovel_grasp.Robot_test import Robot
    if method is None:
        print('Method has to be specified!')
        return
    # Some things to log
    num_of_episode = 0
    num_of_stuck = 0
    test_action_count = 0
    episode_action_count = 0
    grasp_success_count = 0
    # --------------- Setup options ---------------
    obj_root_dir = os.path.abspath(args.obj_root_dir)
    num_obj = 1
    random_seed = args.random_seed
    # Set random seed
    np.random.seed(random_seed)
    rotation_range = 90.
    num_replay_per_step = 2
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
    # Resolution to be used to project action position in voxel coordinate back to the workspace
    action_space_resolution = float(workspace_limits[0][1] - workspace_limits[0][0]) / action_space_size
    # ------------- Algorithm options -------------
    future_reward_discount = args.future_reward_discount
    # ------ Pre-loading and logging options ------
    continue_logging = False
    logging_directory = os.path.abspath('logs')
    save_visualizations = args.save_visualizations  # Save visualizations of FCN predictions?

    # Initialize pick-and-place system (camera and robot)
    robot = Robot(obj_root_dir,
                  num_obj,
                  workspace_limits)
    # Initialize data logger
    logger = Logger(continue_logging,
                    logging_directory,
                    logger_dir=logger_dir)
    logger.save_camera_info(robot.cam_intrinsics, robot.cam_pose, robot.cam_depth_scale)
    logger.save_heightmap_info(workspace_limits, heightmap_resolution)  # Save heightmap parameters

    # Initialize trainer
    if model_logger_dir is not None:
        DQN_model_dir = model_logger_dir
    else:
        DQN_model_dir = None
    trainer = TrainerDQN(action_space_size=action_space_size,
                         future_reward_discount=future_reward_discount,
                         save_model_dir=DQN_model_dir,
                         num_of_rotation=num_rotations,
                         rotation_range=rotation_range)
    if is_testing:
        trainer.shovel_model.eval()
    if not is_testing and continue_iteration > 0:
        trainer.iteration = continue_iteration
        print('Continue With Previous Training')
        print('Check Trainer.iteration: ', trainer.iteration)
        for i in range(trainer.iteration):
            reward = np.load(os.path.join(logger.grasp_success_dir,
                                          '%d.npy' % i))
            trainer.reward_value_log.append(reward)
    # Initialize the Mask-RCNN network (Pretrained)
    device = trainer.device
    mask_rcnn_model = get_model_instance_segmentation(num_classes=2,
                                                      pretrained=False).to(device)
    mask_rcnn_model_path = args.mask_rcnn_model_path
    mask_rcnn_model.load_state_dict(torch.load(mask_rcnn_model_path))
    print('Mask RCNN model preload: ', mask_rcnn_model_path)
    mask_rcnn_model.eval()
    # Initialize the Siamese Network (Pretrained)
    save_model_dir = 'save_file_dir/Siamese_recollect/siamese-99.pth'
    input_img_file_dir = 'datasets/siamese_raw_data_collection'
    anchors_dir = os.path.join(input_img_file_dir, 'anchors')
    classifier_model = siamese_model.ResModel(in_channels=3).to(device)
    classifier_model.load_state_dict(torch.load(save_model_dir))
    classifier_model.train()

    anchor = np.load(os.path.join(anchors_dir, 'anchor.npy'))
    anchor = np.asarray([anchor])
    anchor_input = torch.from_numpy(anchor).to(device, dtype=torch.float)
    anchor_input.requires_grad = False
    anchor_vec = classifier_model.forward(anchor_input).cpu().data.detach().numpy()

    # Initialize variables for heuristic bootstrapping and exploration probability
    no_change_count = 0
    explore_prob = 0.999 if not is_testing else 0.0
    if continue_iteration > 100:
        explore_prob = 0.999 * (0.9 ** continue_iteration)
        print('Continue exploration probability: ', explore_prob)

    # Quick hack for nonlocal memory between threads in Python 2
    nonlocal_variables = {'action_position': None,
                          'rot_angle': 0.,
                          'rot_idx': 0}
    exit_called = False
    while True:
        # Experiment Execution Thread
        if trainer.iteration > 1000:
            num_replay_per_step = 4
        print('\n%s iteration: %d' % ('Training', trainer.iteration))
        iteration_time_0 = time.time()
        if trainer.iteration < 100:
            weight_scale = 1e-1
        else:
            weight_scale = 1

        sim_ok = robot.check_sim()
        if not sim_ok:
            robot.restart_sim()
            episode_action_count = 0
        # Time for the objects to rest so that the image capture does not have problems
        time.sleep(0.5)
        restart_flag = False

        color_heightmap, depth_heightmap = get_heightmaps(robot,
                                                          workspace_limits,
                                                          heightmap_resolution)
        logger.save_npy(color_heightmap,
                        trainer.iteration,
                        logger.color_heightmaps_directory)
        logger.save_npy(depth_heightmap,
                        trainer.iteration,
                        logger.depth_heightmaps_directory)

        # -------- Get corresponding mask and save to file -------
        mask_set, box_set = get_rcnn_output(mask_rcnn_model=mask_rcnn_model,
                                            color_heightmap=color_heightmap,
                                            depth_heightmap=depth_heightmap,
                                            heightmap_size=heightmap_size,
                                            device=device)
        # ===== Generating the Correct mask
        num_of_masks = len(mask_set)
        if num_of_masks > 0:
            # if mask_detection is True and trainer.iteration < 200:
            if mask_detection is True:
                distance_set = np.zeros(num_of_masks)
                for i in range(num_of_masks):
                    vector = compute_siamese_vector(siamese_model=classifier_model,
                                                    color_heightmap=color_heightmap,
                                                    depth_heightmap=depth_heightmap,
                                                    mask=mask_set[i, 0, :, :],
                                                    heightmap_size=heightmap_size,
                                                    device=device)
                    distance = get_distance(anchor_vec, vector)
                    distance_set[i] = distance
                    # ===== Debug plot
                    print('distance= ', distance)
                    # f1 = plt.figure(0)
                    # plt.imshow(color_heightmap)
                    # f2 = plt.figure(1)
                    # plt.imshow(mask_set[i, 0, :, :])
                    # plt.show()
                    # print()
                    # plt.close(f1)
                    # plt.close(f2)
                # ==== Mask Selection
                idx = np.argwhere(distance_set < 2)
                if len(idx) > 0:
                    mask = np.zeros_like(depth_heightmap, dtype=np.float)
                    for i in range(len(idx)):
                        mask = np.add(mask, mask_set[i, 0, :, :])
                else:
                    idx = np.argwhere(distance_set < 4.7)
                    if len(idx) > 0:
                        mask = np.zeros_like(depth_heightmap, dtype=np.float)
                        for i in range(len(mask_set)):
                            mask = np.add(mask, mask_set[i, 0, :, :])
                    else:
                        mask = np.ones_like(depth_heightmap, dtype=np.float)
            else:
                # For non-masked methods, use all-one mask for easy implementation
                mask = np.ones_like(depth_heightmap,
                                    dtype=np.float)
            # ===== Visualize Mask in figure ==
            fig_0 = plt.figure()
            ax_1 = fig_0.add_subplot(1, 2, 1)
            ax_1.imshow(cv2.flip(ndimage.rotate(color_heightmap, 90, reshape=False), 0))
            ax_1.title.set_text('Workspace')
            ax_2 = fig_0.add_subplot(1, 2, 2)
            ax_2.imshow(cv2.flip(ndimage.rotate(mask, 90, reshape=False), 0))
            ax_2.title.set_text('Mask Detection')
            plt.savefig('Mask_vsial.png')
            plt.close(fig_0)
            # ===== Resize the mask to the action space size
            zoom_ratio = float(action_space_size) / heightmap_size
            mask = ndimage.zoom(mask, zoom=[zoom_ratio, zoom_ratio], mode='nearest')
            mask = np.where(mask > 0.2, 1., 0.)

            # ------ Save the updating masks according to different method specifications
            if method == 'DQN':
                update_mask = np.ones_like(mask)
                logger.save_npy(np.ones_like(mask),
                                trainer.iteration,
                                logger.mask_dir)
            elif method == 'DQN_init':
                logger.save_npy(np.ones_like(mask),
                                trainer.iteration,
                                logger.mask_dir)
                if trainer.iteration > 200:
                    update_mask = np.ones_like(mask)
                else:
                    update_mask = mask.copy()
            else:
                update_mask = mask.copy()
                logger.save_npy(update_mask,
                                trainer.iteration,
                                logger.mask_dir)
            kernel = np.ones([3, 3])
            mask = cv2.erode(mask, kernel=kernel, iterations=1)

            # ==== Construct Input to the visual affordance network
            DQN_input_data_image = process_input_imgs(color_heightmap,
                                                      depth_heightmap,
                                                      heightmap_size)  # img-like (H, W, C=4), padded
            if not is_testing and 'prev_shovel_success' in locals():
                # ==== Visual Affordance Update Thread
                # Compute training labels
                label_value, prev_reward_value = trainer.get_label_value(prev_shovel_success,
                                                                         DQN_input_data_image,
                                                                         prev_terminate_state)
                # SGD Backprop
                training_loss = trainer.backprop_mask(prev_DQN_input_data,
                                                      prev_rot_idx,
                                                      prev_action_position,
                                                      action_space_size,
                                                      label_value,
                                                      prev_mask,
                                                      weight_scale=weight_scale)

                print('Last Step Explore Training Loss: %f' % training_loss)
                trainer.training_loss_log.append(training_loss)
                logger.save_npy(np.asarray(trainer.training_loss_log),
                                0,
                                logger.training_loss_dir)

                # Decide Successful Grasp
                if prev_reward_value < 0.1:
                    no_change_count += 1
                else:
                    no_change_count = 0

                print(" ------->>> Experience Replay ----------")
                if trainer.iteration > 20:
                    # ======== Experience Replay Thread
                    positive_samples = np.argwhere(np.asarray(trainer.reward_value_log) > 0.5)
                    # Replay Terminal Transitions
                    if len(positive_samples) > 1:
                        if len(positive_samples) > num_replay_per_step:
                            idx = np.random.choice(len(positive_samples), num_replay_per_step)
                        else:
                            idx = [0]
                        for i in range(len(idx)):
                            _ = experience_replay(sample_iteration=positive_samples[idx[i]][0],
                                                  logger=logger,
                                                  trainer=trainer,
                                                  output_size=action_space_size,
                                                  heightmap_size=heightmap_size,
                                                  weight_scale=weight_scale)
                        # === Replay non-terminal Transitions
                        neg_samples = np.argwhere(np.asarray(trainer.reward_value_log) < 0.5)
                        if len(neg_samples) > 1:
                            if len(neg_samples) > num_replay_per_step:
                                idx = np.random.choice(len(neg_samples), num_replay_per_step)
                            else:
                                idx = [0]
                            for i in range(len(idx)):
                                _ = experience_replay(sample_iteration=neg_samples[idx[i]][0],
                                                      logger=logger,
                                                      trainer=trainer,
                                                      output_size=action_space_size,
                                                      heightmap_size=heightmap_size,
                                                      weight_scale=weight_scale)
                    else:
                        print('Not enough prior training samples. Skipping experience replay.')

                # save Model
                # logger.save_backup_model(trainer.shovel_model, 'shovel')
                if trainer.iteration % 20 == 0:
                    trainer.update_target_net()
                if trainer.iteration % 100 == 0:
                    logger.save_model(trainer.iteration, trainer.shovel_model, 'shovel_DQN_model')
                    trainer.shovel_model.cuda()

            # -------- Enough Training, Let's Act ! ----------
            # Visual Affordance Network Forward inference
            shovel_predictions_raw = trainer.make_predictions(DQN_input_data_image,
                                                              action_space_size)
            # If the prediction should be filtered by the masks
            if filtered_exploration:
                shovel_predictions = np.zeros_like(shovel_predictions_raw)
                for i in range(num_rotations):
                    shovel_predictions[i, :, :] = np.multiply(shovel_predictions_raw[i, :, :],
                                                              mask)
            else:
                shovel_predictions = shovel_predictions_raw

            predicted_value = np.amax(shovel_predictions)
            print('Primitive confidence scores: %f (shovel)' % predicted_value)
            unravel_idx = np.unravel_index(np.argmax(shovel_predictions),
                                           shovel_predictions.shape)
            # ---------------------------------------------------------
            # This is the Exploration Strategy
            print('Exploration Probability: ', explore_prob)
            explore_actions = np.random.uniform() < explore_prob
            explore_prob = explore_prob * 0.99
            if explore_actions and not is_testing:
                print(' >>>>>> Explore Action >>>>>')
                random_rot_idx = np.random.randint(0, num_rotations)
                rand_1 = np.random.randint(-1, 2)
                rand_2 = np.random.randint(-1, 2)
                if unravel_idx[1] + rand_1 < 0 or unravel_idx[1] + rand_1 >= action_space_size:
                    explore_1 = unravel_idx[1]
                else:
                    explore_1 = unravel_idx[1] + rand_1

                if unravel_idx[2] + rand_2 < 0 or unravel_idx[2] + rand_2 >= action_space_size:
                    explore_2 = unravel_idx[2]
                else:
                    explore_2 = unravel_idx[2] + rand_2
                unravel_idx = (random_rot_idx,
                               explore_1,
                               explore_2)
            else:
                print(' <<<<<< Greedy <<<<<< Exploit Action <<<<<')
            # ---------- Exploration Done ---------------

            # Save predicted confidence value
            nonlocal_variables['rot_idx'] = unravel_idx[0]
            nonlocal_variables['rot_angle'] = rotation_range * unravel_idx[0] / (num_rotations-1)  # In Degree
            nonlocal_variables['action_position'] = unravel_idx[1:]  # stores the position in img space

            # ---- Calculate the position that should be sent into the robot ----
            robot_space_x = nonlocal_variables['action_position'][1]
            robot_space_y = nonlocal_variables['action_position'][0]
            x_position = (robot_space_x + 0.5) * action_space_resolution + workspace_limits[0][0]
            y_position = (robot_space_y + 0.5) * action_space_resolution + workspace_limits[1][0]
            # -------- 2D version z calculation -----------------------------
            io_ratio = float(heightmap_size) / action_space_size
            x_low, y_low = int(np.floor(robot_space_x * io_ratio)), int(np.floor(robot_space_y * io_ratio))
            x_high, y_high = int(np.floor((robot_space_x + 1) * io_ratio)), int(np.floor((robot_space_y + 1) * io_ratio))
            local_vox = depth_heightmap[y_low:y_high, x_low:x_high]
            z_position = np.amax(local_vox)
            # ----- Save Action Info
            logger.save_npy(np.asarray(unravel_idx),
                            trainer.iteration,
                            logger.action_dir)   # [rot_idx, img_x, img_y]
            # ---------- Visualize executed primitive, and affordances ----------------
            if save_visualizations:
                shovel_pred_vis = get_prediction_vis(shovel_predictions,
                                                     color_heightmap,
                                                     nonlocal_variables['rot_idx'],
                                                     nonlocal_variables['action_position'],
                                                     num_rotation=num_rotations,
                                                     rotate_angle_range=rotation_range
                                                     )
                logger.save_visualizations(trainer.iteration, shovel_pred_vis, 'shovel')
                cv2.imwrite('visualization.shovel.png', shovel_pred_vis)
                shovel_pred_vis_raw = get_prediction_vis(shovel_predictions_raw,
                                                         color_heightmap,
                                                         nonlocal_variables['rot_idx'],
                                                         nonlocal_variables['action_position'],
                                                         num_rotation=num_rotations,
                                                         rotate_angle_range=rotation_range
                                                         )
                logger.save_visualizations(trainer.iteration,
                                           shovel_pred_vis_raw,
                                           'shovel_raw')
                cv2.imwrite('visualization_raw.shovel.png', shovel_pred_vis_raw)
            # ----- Initialize variables that influence reward
            robot_act_pos = (x_position, y_position, z_position)
            shovel_success = robot.shovel(robot_act_pos,
                                          nonlocal_variables['rot_angle'])
            episode_action_count = episode_action_count + 1
            print('Shovel successful: ', shovel_success)
            if shovel_success:
                if is_testing:
                    grasp_success_count += 1
                trainer.reward_value_log.append(1)
                logger.save_npy(np.asarray(1),
                                trainer.iteration,
                                logger.grasp_success_dir)
                restart_flag = True
            else:
                trainer.reward_value_log.append(0)
                logger.save_npy(np.asarray(0),
                                trainer.iteration,
                                logger.grasp_success_dir)
            # Save information for next training step
            prev_DQN_input_data = DQN_input_data_image.copy()
            prev_rot_idx = nonlocal_variables['rot_idx']
            # prec_rot_angle = nonlocal_variables['rot_angle']
            prev_shovel_success = shovel_success
            prev_action_position = nonlocal_variables['action_position']
            # prev_mask = mask.copy()
            prev_mask = update_mask.copy()
            prev_terminate_state = False

            if len(robot.target_handles) > 0:
                obj_pos = robot.get_single_obj_position(robot.target_handles[0])
                if obj_pos[0] < workspace_limits[0][0] + 0.08 \
                        or obj_pos[0] > workspace_limits[0][1] - 0.08 \
                        or obj_pos[1] < workspace_limits[1][0] + 0.08 \
                        or obj_pos[1] > workspace_limits[1][1] - 0.08:
                    restart_flag = True

            if not prev_shovel_success:
                no_change_count += 1

            if is_testing:
                is_stuck = no_change_count > 15
            else:
                is_stuck = no_change_count > 15

            if is_stuck:
                num_of_stuck += 1
                restart_flag = True

            # Save terminate state information
            if restart_flag is True:
                num_of_episode += 1
                test_action_count += episode_action_count
                episode_action_count = 0
                logger.save_npy(np.asarray(1),
                                trainer.iteration,
                                logger.terminate_state_dir)
                prev_terminate_state = True
                if prev_shovel_success:
                    trainer.reward_episode_log.append(1)
                else:
                    trainer.reward_episode_log.append(0)
            else:
                logger.save_npy(np.asarray(0),
                                trainer.iteration,
                                logger.terminate_state_dir)
                prev_terminate_state = False
            trainer.iteration += 1
            logger.save_npy(np.asarray(trainer.reward_episode_log),
                            0,
                            logger.episode_reward_dir)
        else:
            restart_flag = True
            prev_shovel_success = False

        iteration_time_1 = time.time()

        if restart_flag:
            no_change_count = 0
            is_stuck = False
            print('Long Time no progress/ Object Gone. Restart Simulation.')
            robot.stop_sim()
            robot.restart_sim()
            restart_flag = False

        if is_testing:
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
        else:
            if trainer.iteration >= 8001:
                robot.stop_sim()
                break
        print('Time elapsed: %f' % (iteration_time_1 - iteration_time_0))
        print('Is process terminate: ', prev_terminate_state)
        print('Episode Num: %d' % num_of_episode)
        print('Stuck Number: ', num_of_stuck)
        if is_testing:
            print('Episode action count: ', episode_action_count)
            print('Total action coun: ', test_action_count)
            print('Grasp Success Count: ', grasp_success_count)
            if grasp_success_count > 6:
                return


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
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=True,
                        help='save visualizations of FCN predictions?')
    # Run main program with specified arguments
    args = parser.parse_args()

    # ========= The following code are an example to start a training experiment
    rotation_list = [19]    # The number of rotation angles the visual affordance network you want to train
    method = 'KIDQN'        # The method name
    continue_iteration = 6200  # If you want to continue training from a previous experience
    filtered_exploration = False  # True if you want the robot only explore on the target object detected
    mask_detection = True   # True --> Using the masks detection module; False --> Otherwise
    for num_rotations in rotation_list:
        continue_iteration = continue_iteration
        logger_dir = 'train1/%s/rot%d' % (method, num_rotations)  # Path to save the training logs
        # Path to load pretrained models. If training from scratch, set model_logger_dir = None
        model_logger_dir = 'logs/train1/%s/rot%d/transitions/DQN_models/%d_shovel_DQN_model.pth' % (method,
                                                                                                    num_rotations,
                                                                                                    continue_iteration)
        main(args,
             method=method,
             num_rotations=num_rotations,
             logger_dir=logger_dir,
             model_logger_dir=model_logger_dir,
             is_testing=False,  # The model should be on training mode
             filtered_exploration=filtered_exploration,
             mask_detection=mask_detection,
             continue_iteration=continue_iteration)

    # ========= The following code are examples to start a testing experiment
    # ========= Modify settings accordingl
    # rot_list = [4, 7, 10, 19]  # How many grasp rotation angles you want to consider for the robot action
    # affordance_list = [4, 7, 10, 19]  # The number of rotation angles the visual affordance network is trained
    #                                   # Only affects the log directory thougg
    # model_idx_list = [7500]           # Which model are you testing
    # method = 'KIDQN'                  # Which model method are you testin
    # for model_idx in model_idx_list:
    #     for vis_rot in affordance_list:
    #         for num_rotations in rot_list:
    #         # num_rotations = vis_rot
    #             print('Starting test: %d of %d with model_idx %d' % (num_rotations,
    #                                                                  vis_rot,
    #                                                                  model_idx))
    #             main(args,
    #                  method=method,
    #                  num_rotations=num_rotations,
    #                  logger_dir='test_vis/%s/rot%d_of_%d/%d' % (method,
    #                                                             num_rotations,
    #                                                             vis_rot,
    #                                                             model_idx),
    #                  model_logger_dir='logs/train1/%s/rot%d/transitions/DQN_models/%d_shovel_DQN_model.pth' % (method,
    #                                                                                                            vis_rot,
    #                                                                                                            model_idx),
    #                  is_testing=True,
    #                  filtered_exploration=True,
    #                  mask_detection=True)

