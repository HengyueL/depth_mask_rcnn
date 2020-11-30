import torch, torchvision
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from maskrcnn_training.sd_model import get_transform, get_model_instance_segmentation
from siamese_dataset import Raw_dataset, Siamese_dataset, transform_img, transform_img_rgbd
from siamese_model import ResModel, TripletLoss
# from torchvision.transforms import functional as F


def get_distance(a, b):
    return np.sqrt(np.sum(np.power((a-b), 2)))


def get_torch_input_tenspor(ndarray_data, device):
    input_data = [torch.from_numpy(ndarray_data).float().to(device)]
    return input_data


def get_box_size(box):
    size = min((box[1][0] - box[0][0]), (box[1][1] - box[0][1]))
    return size


def get_mask(img_array, model, device):
    input_array = get_torch_input_tenspor(img_array, device=device)
    output_tensor = model(input_array)[0]
    mask_set = output_tensor['masks'].cpu().data.detach().numpy()
    label_set = output_tensor['labels'].cpu().data.detach().numpy()
    score_set = output_tensor['scores'].cpu().data.detach().numpy()
    # box_set = output_tensor['boxes'].cpu().data.detach().numpy()

    mask = np.zeros_like(img_array[0, :, :])
    iteration_number = score_set.shape[0]
    for i in range(iteration_number):
        if score_set[i] > 0.5 and label_set[i] > 0:
            mask = np.add(mask, mask_set[i, 0, :, :])
    mask = np.where(mask > 0.5, 1., 0.)
    return mask


key_words = ['bowl',
             'cracker_box',
             'lego',
             'mustard',
             'obj_red',
             'sponge',
             'sugar_box']


# ============ Using mask RCNN to detect Masks ======
def derive_masks_for_siamese_data(obj_cls=[]):
    input_img_file_dir = 'datasets/siamese_raw_data_collection'
    output_img_file_dir = 'datasets/siamese_raw_data_collection'
    if not os.path.exists(output_img_file_dir):
        os.mkdir(output_img_file_dir)

    #  Load Pretrained rcnn model
    maskcnn_model_dir = 'save_file_dir/pytorch_sdrcnn_cocoeval2/19.pth'
    rcnn_model = get_model_instance_segmentation(num_classes=2)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    rcnn_model.to(device, dtype=torch.float)
    rcnn_model.load_state_dict(torch.load(maskcnn_model_dir))
    rcnn_model.eval()

    #  Construct raw dataset
    data_set = Raw_dataset(input_img_file_dir,
                           key_words=obj_cls)
    for sample_idx in range(len(data_set)):
        sample = data_set[sample_idx]
        for keyword in obj_cls:
            keyword_data = sample[keyword]
            keyword_mask = get_mask(keyword_data,
                                    model=rcnn_model,
                                    device=device)
            keyword_mask_dir = os.path.join(output_img_file_dir,
                                             keyword + '_mask')
            if not os.path.exists(keyword_mask_dir):
                os.mkdir(keyword_mask_dir)
            save_dir = os.path.join(keyword_mask_dir,
                                    '%d.npy' % sample_idx)
            np.save(save_dir, keyword_mask)

        print('Sample Number: ', sample_idx)
# ============== =======================


# ========== Siamese Network Training Thred ======
def siamese_train_test(is_training=False,
                       obj_cls=[]):
    # ==== Training and Testing Settings ======
    input_img_file_dir = 'datasets/siamese_raw_data_collection'
    save_model_dir = 'save_file_dir/Siamese_ycb'
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)
    if is_training:
        batch_size = 4
        num_epoch = 20
    else:
        batch_size = 1
        num_epoch = 1

    if is_training:
        dataset = Siamese_dataset(input_img_file_dir,
                                  key_words=obj_cls)
    else:
        dataset = Siamese_dataset(input_img_file_dir,
                                  key_words=obj_cls)
        distance_log = {}
        for i in range(len(obj_cls)):
            distance_log[obj_cls[i]] = np.zeros([len(obj_cls),
                                                 len(dataset)])

    siamese_model = ResModel(in_channels=4)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    siamese_model.to(device, dtype=torch.float)
    if is_training:
        siamese_model.train()
        loss_func = TripletLoss(margin=5)
        optimizer = torch.optim.SGD(siamese_model.parameters(),
                                    lr=1e-4, momentum=0.9,
                                    weight_decay=2e-5)
    else:
        siamese_model.train()
        siamese_model.load_state_dict(torch.load(os.path.join(save_model_dir, 'siamese-19.pth')))
    if is_training:
        # n_batch = len(dataset) // batch_size
        n_batch = 15
    else:
        n_batch = 100
    epoch_loss_log = []
    loss_log = []

    for epoch in range(num_epoch):
        positive_idx_set = np.arange(0, len(dataset))
        negative_idx_set = np.arange(0, len(dataset))
        np.random.shuffle(positive_idx_set)
        np.random.shuffle(negative_idx_set)

        print("Epoch : %d" % epoch)
        epoch_loss = 0
        anchor_idx = np.random.choice(len(dataset), batch_size)
        print("Random Anchor Idx : ", anchor_idx)
        for batch_idx in range(n_batch):
            print('Training Batch: %d of %d batches' % (batch_idx, n_batch))
            anchor_batch = []
            positive_batch = []
            negative_batch = []
            for key_number in range(len(obj_cls)):
                for negative_number in range(len(obj_cls)):
                    if negative_number != key_number:
                        pos_cls = obj_cls[key_number]
                        neg_cls = obj_cls[negative_number]
                        print('Anchoring: ', pos_cls)
                        print('Against: ', neg_cls)
                        print(' =========================== ')
                        for i in range(batch_size):
                            anchor_data_check = np.amax(dataset[anchor_idx[i]][pos_cls][0, :, :]) > 0
                            pos_data_check = np.amax(dataset[positive_idx_set[i + batch_idx * batch_size]][pos_cls][0, :, :]) > 0
                            neg_data_check = np.amax(dataset[negative_idx_set[i + batch_idx * batch_size]][neg_cls][0, :, :]) > 0
                            if anchor_data_check and pos_data_check and neg_data_check:
                                anchor_batch.append(dataset[anchor_idx[i]][pos_cls])
                                positive_batch.append(dataset[positive_idx_set[i + batch_idx * batch_size]][pos_cls])
                                negative_batch.append(dataset[negative_idx_set[i + batch_idx * batch_size]][neg_cls])
                        if len(anchor_batch) > 0:
                            anchor_batch = np.asarray(anchor_batch)
                            positive_batch = np.asarray(positive_batch)
                            negative_batch = np.asarray(negative_batch)
                            if is_training:
                                anchor_data = torch.as_tensor(anchor_batch).to(device, dtype=torch.float)
                                positive_data = torch.as_tensor(positive_batch).to(device, dtype=torch.float)
                                neg_data = torch.as_tensor(negative_batch).to(device, dtype=torch.float)
                                anchor_data.requires_grad = False
                                positive_data.requires_grad = True
                                neg_data.requires_grad = False
                                optimizer.zero_grad()
                            else:
                                anchor_data = torch.as_tensor(anchor_batch).to(device, dtype=torch.float)
                                positive_data = torch.as_tensor(positive_batch).to(device, dtype=torch.float)
                                neg_data = torch.as_tensor(negative_batch).to(device, dtype=torch.float)
                                anchor_data.requires_grad = False
                                positive_data.requires_grad = False
                                neg_data.requires_grad = False

                            positive_pred = siamese_model(positive_data)
                            neg_pred = siamese_model(neg_data)
                            anchor_pred = siamese_model(anchor_data)
                            if is_training:
                                loss = loss_func(anchor_pred, positive_pred, neg_pred)
                                loss.backward()
                                optimizer.step()
                                loss_value = loss.cpu().data.detach().numpy()
                                loss_log.append(loss_value)
                                epoch_loss += loss_value

                            else:
                                anchor_pred_data = anchor_pred.cpu().data.numpy()
                                positive_pred_data = positive_pred.cpu().data.numpy()
                                neg_pred_data = neg_pred.cpu().data.numpy()
                                pos_distance = get_distance(anchor_pred_data,
                                                            positive_pred_data)
                                distance_log[obj_cls[key_number]][key_number, batch_idx] = pos_distance
                                neg_distance = get_distance(neg_pred_data,
                                                            anchor_pred_data)
                                distance_log[obj_cls[key_number]][negative_number, batch_idx] = neg_distance
                            anchor_batch = []
                            positive_batch = []
                            negative_batch = []
                # Refresh loss visualization for every key object
                fig_1 = plt.figure()
                ax_1 = fig_1.add_subplot(2, 1, 1)
                ax_2 = fig_1.add_subplot(2, 1, 2)
                ax_1.plot(loss_log)
                ax_1.title.set_text('Step Loss')
                ax_2.plot(epoch_loss_log)
                ax_2.title.set_text('Epoch Accumulative Loss')
                plt.savefig('siamese-training-loss.png')
                plt.close(fig_1)
        if is_training:
            epoch_loss_log.append(epoch_loss)
            torch.save(siamese_model.state_dict(),
                       os.path.join(save_model_dir, 'siamese-%d.pth' % epoch))
        else:
            distance_save_dir = os.path.join(save_model_dir, 'distance_log')
            if not os.path.exists(distance_save_dir):
                os.mkdir(distance_save_dir)
            for key in distance_log.keys():
                save_name = os.path.join(distance_save_dir, key + '_log')
                np.save(save_name,
                        distance_log[key])


def visualize_simese_result(log_dir=None):
    # testlog_dir = 'save_file_dir/Siamese_ycb/distance_log'
    if log_dir is not None:
        testlog_dir = log_dir
        test_log_files = [f for f in os.listdir(testlog_dir) if
                          os.path.isfile(os.path.join(testlog_dir, f)) and ('.npy' in f)]
        test_log_files.sort()
        for i in range(len(test_log_files)):
            data = np.load(os.path.join(testlog_dir,
                                        test_log_files[i]),
                           allow_pickle=True)
            fig_1 = plt.figure()
            ax_1 = fig_1.add_subplot(1, 1, 1)
            for j in range(data.shape[0]):
                f = ax_1.plot(data[j, 0:100], label='%s' % test_log_files[j][:-8])
            ax_1.legend()
            ax_1.title.set_text('Anchor: %s' % test_log_files[i][:-8])
            plt.savefig(os.path.join(testlog_dir,
                                     'siamese_distance_%s.png' % test_log_files[i][:-8]))
            plt.close(fig_1)
    else:
        print('Invalid log dir input.')


def visualize_dataset(obj_cls=[]):
    # ==== Training and Testing Settings ======
    input_img_file_dir = 'datasets/siamese_raw_data_collection'
    dataset = Siamese_dataset(input_img_file_dir,
                              key_words=obj_cls)
    for i in range(len(dataset)):
        fig_1 = plt.figure(1)
        for j in range(len(obj_cls)):
            data = dataset[i][obj_cls[j]]
            data_visual = np.transpose(data, [1, 2, 0])
            ax = fig_1.add_subplot(1, len(obj_cls), j+1)
            ax.imshow(data_visual)
        plt.show()
        plt.close(fig_1)



# # ==== Construct Mask from raw images ==== #
# derive_masks_for_siamese_data(obj_cls=key_words)

# # ==== Training and Testing Thred ==== #
# siamese_train_test(is_training=True,
#                    obj_cls=key_words)
# siamese_train_test(is_training=False,
#                    obj_cls=key_words)
# print('That\'s it!')

# # ======= Visualize Training Result ====== #
# testlog_dir = 'save_file_dir/Siamese_ycb/distance_log'
# visualize_simese_result(log_dir=testlog_dir)

# ======
# visualize_dataset(obj_cls=key_words)
# ======


def moving_avg(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n

# rot_list = [3, 4, 7, 10, 19]
# vis_list = [4, 7, 10, 19]
# action_eff_log = []
# for vis_idx in vis_list:
#     for rot_idx in rot_list:
#         dir = 'logs/test/KIDQN/rot%d_of_%d' % (rot_idx, vis_idx)
#         data_dir = os.path.join(dir, 'transitions/episode-reward-log/0.npy')
#         data = np.load(data_dir)
#         data_sum = moving_avg(data, n=10)
#         num_episode = len(data)
#         total_reward = np.sum(data)
#         total_iteration = 100
#         # data_sum = np.zeros_like(data)
#         # data_sum[0] = data[0]
#         # for i in range(1, len(data)):
#         #     data_sum[i] = data[i] + data_sum[i-1]
#         # plt.plot(data_sum)
#         print("Episode Number: ", num_episode)
#         print('Episode Success Rate:  ', total_reward / num_episode)
#         print('Action Efficiency:  ', total_reward / total_iteration)
#         action_eff_log.append(total_reward / total_iteration)
# plt.figure()
# plt.plot(rot_list, action_eff_log)
# print()
# ====  print training progress =====
# model_idx_set = list(range(400, 1800, 200))
# # for i in range(1000, 1600, 200):
# #     model_idx_set.append(i)
# # model_idx_set = [6400]
# # model_idx_set = []
# # test_rot_list = [19]
# vis_list = [4, 7, 10, 19]
# # vis_list = [19]
# fig_0 = plt.figure(0)
# ax_0 = fig_0.add_subplot(2, 1, 1)
# ax_1 = fig_0.add_subplot(2, 1, 2)
# # ax_2 = fig_0.add_subplot(4, 1, 3)
# # ax_3 = fig_0.add_subplot(4, 1, 4)
#
# for method_idx in vis_list:
#     # for test_rot in test_rot_list:
#     test_rot = method_idx
#
#     suc_rate_log = []
#     suc_rate_normalized = []
#     training_step_log = []
#     action_efficiency_log = []
#     stuck_num_log = []
#     no_stuck_action_log = []
#     for model_idx in model_idx_set:
#         dir = 'logs/test/KIDQN_progress2/rot%d_of_%d/%d' % (test_rot,
#                                                             method_idx,
#                                                             model_idx)
#         data_dir = os.path.join(dir, 'transitions/episode-reward-log/0.npy')
#         stuck_num_dir = os.path.join(dir, 'transitions/0.npy')
#         episode_num_dir = os.path.join(dir, 'transitions/1.npy')
#         action_count_dir = os.path.join(dir, 'transitions/2.npy')
#         data = np.load(data_dir)
#         stuck_num = np.load(stuck_num_dir)
#         episode_num = np.load(episode_num_dir)
#         action_count = np.load(action_count_dir)
#
#         suc_rate_log.append(np.sum(data) / episode_num)
#         suc_rate_normalized.append(np.sum(data) / (episode_num-stuck_num))
#         training_step_log.append(model_idx * 10 * 2)
#         action_efficiency_log.append(np.sum(data) / action_count)
#         stuck_num_log.append(stuck_num)
#         no_stuck_action_log.append(action_count - 10*stuck_num)
#
#     suc_rate_log = np.asarray(suc_rate_log)
#     suc_rate_normalized = np.asarray(suc_rate_normalized)
#     training_step_log = np.asarray(training_step_log)
#     action_efficiency_log = np.asarray(action_efficiency_log)
#     stuck_num_log = np.asarray(stuck_num_log)
#     no_stuck_action_log = np.asarray(no_stuck_action_log)
#     ax_0.plot(training_step_log,
#               suc_rate_log,
#               label='%d' % method_idx)
#     # ax_1.plot(training_step_log,
#     #           suc_rate_normalized,
#     #           label='%d' % method_idx)
#     ax_1.plot(training_step_log,
#               action_efficiency_log,
#               label='%d' % method_idx)
#     # ax_3.plot(training_step_log,
    #           stuck_num_log,
    #           label='%d' % method_idx)
        # ax_3.plot(training_step_log,
        #           no_stuck_action_log,
        #           label='%d' % method_idx)

# ax_0.title.set_text('Suc Rate')
# ax_0.grid()
# ax_0.legend()
# ax_0.set_xlabel('num of training steps')
# ax_1.title.set_text('Action Efficiency')
# ax_1.grid()
# ax_1.legend()
# ax_2.title.set_text('Action efficiency')
# ax_3.title.set_text('Stuck episodes')
# ax_2.legend()
# ax_3.legend()

# base_line_root_dir = 'logs/test/baselines'
# baseline_method = ['human']
# # baseline_method = ['human', 'random']
# # data_len = len(training_step_log)
# for method in baseline_method:
#     dir = os.path.join(base_line_root_dir, method)
#     data_dir = os.path.join(dir, 'transitions/episode-reward-log/0.npy')
#     stuck_num_dir = os.path.join(dir, 'transitions/0.npy')
#     episode_num_dir = os.path.join(dir, 'transitions/1.npy')
#     action_count_dir = os.path.join(dir, 'transitions/2.npy')
#     data = np.load(data_dir)
#     stuck_num = np.load(stuck_num_dir)
#     episode_num = np.load(episode_num_dir)
#     action_count = np.load(action_count_dir)
#
#     suc_rate_log = np.ones_like(training_step_log) * (np.sum(data) / episode_num)
#     # suc_rate_normalized.append(np.sum(data) / (episode_num - stuck_num))
#     # training_step_log.append(model_idx * 10 * 2)
#     action_efficiency_log = np.ones_like(training_step_log) * (np.sum(data) / action_count)
#     # stuck_num_log.append(stuck_num)
#     # no_stuck_action_log.append(action_count - 10 * stuck_num)
#     ax_0.plot(training_step_log,
#               suc_rate_log,
#               '--',
#               label='%s' % method)
#     ax_1.plot(training_step_log,
#               action_efficiency_log,
#               label='%s' % method)
# ax_0.set_ylim(0.0, 1.0)
# ax_0.title.set_text('Suc Rate')
# ax_0.grid()
# ax_0.legend()
# ax_0.set_xlabel('num of training steps')
# ax_1.set_ylim(-0.1, 0.3)
# ax_1.title.set_text('Action Efficiency')
# ax_1.grid()
# ax_1.legend()
#
# plt.show()
# print()
