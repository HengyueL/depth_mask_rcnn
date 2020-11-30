import time
import os
import numpy as np
from shovel_grasp.trainer import TrainerDQN
from shovel_grasp.logger import Logger
import matplotlib.pyplot as plt
from main_kidqn import experience_replay, process_input_imgs
from shovel_grasp.trainer import get_prediction_vis
import torch
import cv2


def visualize(sample_iteration,
              logger,
              trainer,
              output_size,
              heightmap_size):
    sample_color_img = np.load(os.path.join(logger.color_heightmaps_directory,
                                            '%d.npy' % sample_iteration))
    sample_depth_img = np.load(os.path.join(logger.depth_heightmaps_directory,
                                            '%d.npy' % sample_iteration))
    sample_input_data = process_input_imgs(sample_color_img,
                                           sample_depth_img,
                                           heightmap_size)
    q_values = trainer.make_predictions(sample_input_data,
                                        output_size,
                                        requires_grad=False,
                                        mode='predict')
    unravel_idx = np.unravel_index(np.argmax(q_values),
                                   q_values.shape)
    rot_idx = unravel_idx[0]
    act_position = unravel_idx[1:]
    q_vis = get_prediction_vis(q_values,
                               sample_color_img,
                               rot_idx,
                               act_position,
                               num_rotation=trainer.num_of_rotation,
                               rotate_angle_range=trainer.rotation_range)
    cv2.imwrite('offline_prediction_vis.png',
                q_vis)


def main(num_training_steps=1000,
         model_logger_dir=None,
         num_rotations=7):
    """
    This script is only to train Q maps offline.
    Nothing includes
    """
    continue_logging = True
    logging_directory = 'logs/train/KIDQN/rot%d' % num_rotations
    num_reply_per_step = 5
    num_samples = 4999
    # logger_dir = 'logs/train/offline/KIDQN/rot%d' % num_rotations
    logger_dir = 0

    # ==== Robot Space Params
    future_reward_discount = 0.95
    rotation_range = 90.
    heightmap_size = 500
    action_space_size = 64

    # Initialize data logger
    logger = Logger(continue_logging,
                    logging_directory,
                    logger_dir=logger_dir)
    # DQN training wrapper
    trainer = TrainerDQN(action_space_size=action_space_size,
                         future_reward_discount=future_reward_discount,
                         save_model_dir=model_logger_dir,
                         num_of_rotation=num_rotations,
                         rotation_range=rotation_range)
    offline_model_save_path = os.path.join(logger.transitions_directory,
                                           'offline_model')
    if not os.path.exists(offline_model_save_path):
        os.mkdir(offline_model_save_path)

    trainer.iteration = num_samples
    for i in range(trainer.iteration):
        reward = np.load(os.path.join(logger.grasp_success_dir,
                                      '%d.npy' % i))
        trainer.reward_value_log.append(reward)

    positive_samples = np.argwhere(np.asarray(trainer.reward_value_log) > 0.5)
    neg_samples = np.argwhere(np.asarray(trainer.reward_value_log) < 0.5)
    sample_summary_path = os.path.join(logger.base_directory, 'pos_neg_samples.npy')
    np.save(sample_summary_path,
            np.asarray([len(positive_samples),
                        len(neg_samples)]))

    assert len(positive_samples) > 100, 'Not enough positive grasp samples'
    assert len(neg_samples) > 100, 'Not enough negative grasp samples'
    step_loss_log = []
    for train_step in range(num_training_steps):
        print('Training Step: ', train_step)
        # experience_replay script
        pos_idx = np.random.choice(len(positive_samples), num_reply_per_step)
        neg_idx = np.random.choice(len(neg_samples), num_reply_per_step)

        step_loss = 0
        if train_step < 200:
            weight_scale = 1e-1
        else:
            weight_scale = 1
        for i in range(num_reply_per_step):
            loss1 = experience_replay(sample_iteration=positive_samples[pos_idx[i]][0],
                                      logger=logger,
                                      trainer=trainer,
                                      output_size=action_space_size,
                                      heightmap_size=heightmap_size,
                                      weight_scale=weight_scale)
            step_loss += loss1
            loss2 = experience_replay(sample_iteration=neg_samples[neg_idx[i]][0],
                                      logger=logger,
                                      trainer=trainer,
                                      output_size=action_space_size,
                                      heightmap_size=heightmap_size,
                                      weight_scale=weight_scale)
            step_loss += loss2
        step_loss_log.append(step_loss)
        logger.save_npy(np.asarray(step_loss_log),
                        0,
                        logger.training_loss_dir)
        # === Update Eval Model ===
        if train_step % 20 == 0:
            trainer.update_target_net()
        if train_step % 100 == 0:
            torch.save(trainer.shovel_model.state_dict(),
                       os.path.join(offline_model_save_path,
                                    '%d_%s.pth' % (train_step,
                                                   'offline_DQN')))

        # For visualization
        if train_step % 20 == 0:
            fig_0 = plt.figure()
            ax1 = fig_0.add_subplot(1, 1, 1)
            ax1.plot(step_loss_log)
            ax1.title.set_text('Training Loss (per step (10 backprops) )')
            plt.savefig('Offline_training_loss.png')
            plt.close(fig_0)

            visualize(sample_iteration=positive_samples[pos_idx[0]][0],
                      logger=logger,
                      trainer=trainer,
                      output_size=action_space_size,
                      heightmap_size=heightmap_size)
            print()


if __name__ == '__main__':
    rotation_list = [7, 10, 19]
    for i in rotation_list:
        main(num_training_steps=7000,
             model_logger_dir=None,
             num_rotations=i)
