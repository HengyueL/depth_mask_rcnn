#!/usr/bin/env python
# Main Script of KI-DQN project

import time
import os
import argparse
import numpy as np
import cv2
from scipy import ndimage
from shovel_grasp.Robot import Robot
from shovel_grasp.trainer import TrainerDQN
from shovel_grasp.trainer import get_prediction_vis
from shovel_grasp.logger import Logger
import shovel_grasp.utils as utils
import matplotlib.pyplot as plt
from maskrcnn_training.sd_model import get_model_instance_segmentation
import siamese_model
import torch


def get_distance(a, b):
    return np.sqrt(np.sum(np.power((a-b), 2)))


def get_rcnn_output(mask_rcnn_model,
                    color_heightmap,
                    depth_heightmap,
                    heightmap_size,
                    device):
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
    """
    img_rgb = (255 * img[:, :, 0:3]).astype(np.uint8)
    img_gray = (cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) / 255.).astype(np.float)
    img_depth = img[:, :, 3]
    # depth_max = np.amax(img_depth)
    if is_wall:
        depth_max = 1.
    else:
        depth_max = 0.2

    img_depth = img_depth / depth_max
    x, y = img_depth.shape
    img_depth.shape = (x, y, 1)
    img_gray.shape = (x, y, 1)
    target_img = np.concatenate((img_gray, img_gray, img_depth),
                                axis=2)
    target_img = np.transpose(target_img, [2, 0, 1])
    return target_img


def process_input_imgs(color_heightmap, depth_heightmap, heightmap_size):
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
                      output_size, heightmap_size):
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

    sample_grasp_score, _ = trainer.get_label_value(sample_grasp_success,
                                                    sample_next_state_data)
    loss_value = trainer.backprop_mask(input_img=sample_input_data,
                                       rot_idx=sample_action_position[0],
                                       action_position=sample_action_position[1:3],
                                       output_size=output_size,
                                       label_value=sample_grasp_score,
                                       prev_mask=sample_mask)
    print('Replay Training Loss: %f' % loss_value)


def get_heightmaps(robot, workspace_limits, heightmap_resolution):
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
         is_testing=False,
         filtered_exploration=False):
    # --------------- Setup options ---------------
    obj_root_dir = os.path.abspath(args.obj_root_dir)
    num_obj = args.num_obj
    random_seed = args.random_seed
    # Set random seed
    np.random.seed(random_seed)

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
    # check heightmap resolution
    # Resolution to be used to project action position in voxel coordinate back to the workspace
    action_space_resolution = float(workspace_limits[0][1] - workspace_limits[0][0]) / action_space_size

    # ------------- Algorithm options -------------
    future_reward_discount = args.future_reward_discount

    # ------ Pre-loading and logging options ------
    continue_logging = args.continue_logging  # Continue logging from previous session
    logging_directory = os.path.abspath(args.logging_directory) if continue_logging else os.path.abspath('logs')
    save_visualizations = args.save_visualizations  # Save visualizations of FCN predictions?

    # Initialize pick-and-place system (camera and robot)
    robot = Robot(obj_root_dir,
                  num_obj,
                  workspace_limits)

    # robot.add_wall()
    # position = robot.add_target()
    # robot.shovel(position,
    #              rotation_angle=0.)

    # Initialize data logger
    logger = Logger(continue_logging,
                    logging_directory,
                    logger_dir='KIDQN_training_1')
    logger.save_camera_info(robot.cam_intrinsics, robot.cam_pose, robot.cam_depth_scale)
    logger.save_heightmap_info(workspace_limits, heightmap_resolution)  # Save heightmap parameters

    # Initialize trainer
    trainer = TrainerDQN(action_space_size=action_space_size,
                         future_reward_discount=future_reward_discount,
                         save_model_dir=None)

    device = trainer.device
    mask_rcnn_model = get_model_instance_segmentation(num_classes=2,
                                                      pretrained=False).to(device)
    mask_rcnn_model_path = args.mask_rcnn_model_path
    mask_rcnn_model.load_state_dict(torch.load(mask_rcnn_model_path))
    print('Mask RCNN model preload: ', mask_rcnn_model_path)
    save_model_dir = 'save_file_dir/Siamese_recollect/siamese-99.pth'
    mask_rcnn_model.eval()

    input_img_file_dir = 'datasets/siamese_raw_data_collection'
    anchors_dir = os.path.join(input_img_file_dir, 'anchors')
    classifier_model = siamese_model.ResModel(in_channels=3).to(device)
    classifier_model.load_state_dict(torch.load(save_model_dir))
    classifier_model.train()
    target_anchor = np.load(os.path.join(anchors_dir, 'anchor.npy'))
    anchor_mask = np.load(os.path.join(anchors_dir, 'anchor_mask.npy'))
    anchor = np.zeros_like(target_anchor)
    for i in range(anchor.shape[2]):
        anchor[:, :, i] = np.multiply(target_anchor[:, :, i], anchor_mask)
    # Check Anchor
    anchor = transform_img(anchor)
    anchor = np.asarray([anchor])
    anchor_input = torch.from_numpy(anchor).to(device, dtype=torch.float)
    anchor_input.requires_grad = False
    anchor_vec = classifier_model.forward(anchor_input).cpu().data.detach().numpy()
    print()


    # Initialize variables for heuristic bootstrapping and exploration probability
    no_change_count = 0
    explore_prob = 0.999 if not is_testing else 0.0

    # Quick hack for nonlocal memory between threads in Python 2
    nonlocal_variables = {'action_position': None,
                          'rot_angle': 0.,
                          'rot_idx': 0}
    exit_called = False
    restart_flag = False
    while True:
        if trainer.iteration > 1000:
            num_replay_per_step = 4
        print('\n%s iteration: %d' % ('Training', trainer.iteration))
        iteration_time_0 = time.time()

        sim_ok = robot.check_sim()
        if not sim_ok:
            robot.restart_sim()

        time.sleep(0.5)
        color_heightmap, depth_heightmap = get_heightmaps(robot,
                                                          workspace_limits,
                                                          heightmap_resolution)
        logger.save_npy(color_heightmap,
                        trainer.iteration,
                        logger.color_heightmaps_directory)
        logger.save_npy(depth_heightmap,
                        trainer.iteration,
                        logger.depth_heightmaps_directory)

        # -------------- TO DO: get corresponding mask and save to file -------
        mask_set, box_set = get_rcnn_output(mask_rcnn_model=mask_rcnn_model,
                                            color_heightmap=color_heightmap,
                                            depth_heightmap=depth_heightmap,
                                            heightmap_size=heightmap_size,
                                            device=device)
        # To Do: ===== Generating the Correct mask
        num_of_masks = len(mask_set)
        if num_of_masks > 0:
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
            idx = np.argwhere(distance_set < 3)
            if len(idx) > 0:
                i = np.argmin(distance_set)
                mask = mask_set[i, 0, :, :]
            else:
                idx = np.argwhere(distance_set < 4.7)
                if len(idx) > 0:
                    mask = np.zeros_like(depth_heightmap, dtype=np.float)
                    for i in range(len(mask_set)):
                        mask = np.add(mask, mask_set[i, 0, :, :])
                else:
                    mask = np.ones_like(depth_heightmap, dtype=np.float)
            # ===== Visualize Mask in figure ==
            fig_0 = plt.figure()
            ax_1 = fig_0.add_subplot(1, 2, 1)
            ax_1.imshow(color_heightmap)
            ax_1.title.set_text('Workspace')
            ax_2 = fig_0.add_subplot(1, 2, 2)
            ax_2.imshow(mask)
            ax_2.title.set_text('Mask Detection')
            plt.savefig('Mask_vsial.png')
            plt.close(fig_0)
            # =====================================
            zoom_ratio = float(action_space_size) / heightmap_size
            mask = ndimage.zoom(mask, zoom=[zoom_ratio, zoom_ratio], mode='nearest')
            mask = np.where(mask > 0.2, 1., 0.)
            logger.save_npy(mask,
                            trainer.iteration,
                            logger.mask_dir)
            # --------------------------------------------------------------
            DQN_input_data_image = process_input_imgs(color_heightmap,
                                                      depth_heightmap,
                                                      heightmap_size)  # img-like (H, W, C=4), padded
            # check DQN_input_data_image
            if 'prev_shovel_success' in locals():
                # ------------- Stochastic Gradient Descent ------------------
                # Compute training labels
                label_value, prev_reward_value = trainer.get_label_value(prev_shovel_success,
                                                                         DQN_input_data_image)
                # Backpropagate
                training_loss = trainer.backprop_mask(prev_DQN_input_data,
                                                      prev_rot_idx,
                                                      prev_action_position,
                                                      action_space_size,
                                                      label_value,
                                                      prev_mask)

                print('Last Step Explore Training Loss: %f' % training_loss)
                trainer.training_loss_log.append(training_loss)
                logger.save_npy(np.asarray(trainer.training_loss_log),
                                0,
                                logger.training_loss_dir)

                # Detect changes
                if prev_reward_value < 0.1:
                    no_change_count += 1
                else:
                    no_change_count = 0

                print(" ------->>> Experience Replay ----------")
                if trainer.iteration > 20:
                    # ======== Experience Replay
                    positive_samples = np.argwhere(np.asarray(trainer.reward_value_log) > 0.5)
                    if len(positive_samples) > 1:
                        if len(positive_samples) > num_replay_per_step:
                            idx = np.random.choice(len(positive_samples), num_replay_per_step)
                        else:
                            idx = [0]
                        for i in range(len(idx)):
                            experience_replay(sample_iteration=positive_samples[idx[i]][0],
                                              logger=logger,
                                              trainer=trainer,
                                              output_size=action_space_size,
                                              heightmap_size=heightmap_size)
                        # === Replay Normal Samples
                        neg_samples = np.argwhere(np.asarray(trainer.reward_value_log) < 0.5)
                        if len(neg_samples) > 1:
                            if len(neg_samples) > num_replay_per_step:
                                idx = np.random.choice(len(neg_samples), num_replay_per_step)
                            else:
                                idx = [0]
                            for i in range(len(idx)):
                                experience_replay(sample_iteration=neg_samples[idx[i]][0],
                                                  logger=logger,
                                                  trainer=trainer,
                                                  output_size=action_space_size,
                                                  heightmap_size=heightmap_size)
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
            shovel_predictions_raw = trainer.make_predictions(DQN_input_data_image,
                                                              action_space_size)
            if filtered_exploration:
                shovel_predictions = np.zeros_like(shovel_predictions_raw)
                for i in range(3):
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
            if explore_actions:
                print(' >>>>>> Explore Action >>>>>')
                random_rot_idx = np.random.randint(0, 3)
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
            nonlocal_variables['rot_angle'] = unravel_idx[0] * 45.  # In Degree
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
                # shovel_pred_vis = get_prediction_vis(shovel_predictions,
                #                                      color_heightmap,
                #                                      nonlocal_variables['rot_idx'],
                #                                      nonlocal_variables['action_position'])
                # logger.save_visualizations(trainer.iteration, shovel_pred_vis, 'shovel')
                # cv2.imwrite('visualization.shovel.png', shovel_pred_vis)
                shovel_pred_vis_raw = get_prediction_vis(shovel_predictions_raw,
                                                         color_heightmap,
                                                         nonlocal_variables['rot_idx'],
                                                         nonlocal_variables['action_position'])
                logger.save_visualizations(trainer.iteration, shovel_pred_vis_raw, 'shovel_raw')
                cv2.imwrite('visualization_raw.shovel.png', shovel_pred_vis_raw)
                if trainer.iteration % 10 == 0:
                    fig_1 = plt.figure()
                    ax_1 = fig_1.add_subplot(1, 1, 1)
                    ax_1.plot(trainer.training_loss_log)
                    ax_1.title.set_text('Training Loss')
                    plt.savefig('DQN-training-loss.png')
                    plt.close(fig_1)

            # ----- Initialize variables that influence reward
            # nonlocal_variables['shovel_success'] = False
            robot_act_pos = (x_position, y_position, z_position)
            shovel_success = robot.shovel(robot_act_pos,
                                          nonlocal_variables['rot_angle'])
            print('Shovel successful: ', shovel_success)
            if shovel_success:
                trainer.reward_value_log.append(1)
                logger.save_npy(np.asarray(1),
                                trainer.iteration,
                                logger.grasp_success_dir)
            else:
                trainer.reward_value_log.append(0)
                logger.save_npy(np.asarray(0),
                                trainer.iteration,
                                logger.grasp_success_dir)
            # Save information for next training step
            prev_DQN_input_data = DQN_input_data_image.copy()
            prev_rot_idx = nonlocal_variables['rot_idx']
            prev_shovel_success = shovel_success
            prev_action_position = nonlocal_variables['action_position']
            prev_mask = mask.copy()

            if len(robot.target_handles) > 0:
                obj_pos = robot.get_single_obj_position(robot.target_handles[0])
                if obj_pos[0] < workspace_limits[0][0] - 0.02 \
                        or obj_pos[0] > workspace_limits[0][1] + 0.02 \
                        or obj_pos[1] < workspace_limits[1][0] - 0.02 \
                        or obj_pos[1] > workspace_limits[1][1] + 0.02:
                    restart_flag = True

            if not prev_shovel_success:
                no_change_count += 1
            trainer.iteration += 1
        else:
            restart_flag = True
            prev_shovel_success = False

        iteration_time_1 = time.time()
        if prev_shovel_success or restart_flag or no_change_count > 15:
            no_change_count = 0
            print('Long Time no progress/ Object Gone. Restart Simulation.')
            robot.stop_sim()
            robot.restart_sim()
            restart_flag = False
        if trainer.iteration >= 10000:
            robot.stop_sim()
            break
        print('Time elapsed: %f' % (iteration_time_1 - iteration_time_0))


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--obj_root_dir', dest='obj_root_dir', action='store', default='shovel_grasp/objects',
                        help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--mask_rcnn_model_path', dest='mask_rcnn_model_path', action='store',
                        default='save_file_dir/pytorch_sdrcnn_cocoeval2/19.pth',
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
    main(args, filtered_exploration=True)
