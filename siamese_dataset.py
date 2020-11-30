# Old script of Simese Dataset Construction
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2


def transform_img(img, is_wall=False):
    img_rgb = img[:, :, 0:3].astype(np.uint8)
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
    return target_img


def transform_img_rgbd(img, is_wall=False):
    img_rgb = (img[:, :, 0:3]/255.).astype(np.float)
    # img_mean = [0.485, 0.456, 0.406]
    # img_std = [0.229, 0.224, 0.225]
    # for c in range(3):
    #     img_rgb[:, :, c] = (img_rgb[:, :, c] - img_mean[c]) / img_std[c]
    img_depth = img[:, :, 3]
    # depth_max = np.amax(img_depth)
    if is_wall:
        depth_max = 1.
    else:
        depth_max = 0.2
    img_depth = img_depth / depth_max
    x, y = img_depth.shape
    img_depth.shape = (x, y, 1)
    target_img = np.concatenate((img_rgb, img_depth),
                                axis=2)
    return target_img


def get_raw_data(dir, idx):
    raw_np_array = np.load(dir + '/%d.npy' % idx)
    img = transform_img(raw_np_array)
    img_input = np.transpose(img, [2, 0, 1])
    return img_input


def get_rgbd_data(array_dir,
                  mask_dir,
                  idx):
    raw_np_array = np.load(array_dir + '/%d.npy' % idx)
    mask = np.load(mask_dir + '/%d.npy' % idx)
    for i in range(raw_np_array.shape[2]):
        raw_np_array[:, :, i] = np.multiply(raw_np_array[:, :, i],
                                            mask)
    img = transform_img_rgbd(raw_np_array)
    img_input = np.transpose(img, [2, 0, 1])
    return img_input


class Raw_dataset(Dataset):
    """ Construt Displacement data set for training purpose"""
    def __init__(self, base_dir,
                 key_words=[],
                 transform=None):
        self.base_dir = base_dir
        # ====== Original =========
        # self.red_dir = os.path.join(self.base_dir, 'obj_red_img')
        # self.cheeze_dir = os.path.join(self.base_dir, 'obj_cheeze_img')
        # self.disturb_dir = os.path.join(self.base_dir, 'disturb_img')
        # self.target_obj_dir = os.path.join(self.base_dir, 'obj_img')
        self.wall_dir = os.path.join(self.base_dir, 'wall')

        # ====== YCB ========
        self.key_words = key_words

        # ===== Common ======
        self.transform = transform

    def __len__(self):
        files_log = [f for f in os.listdir(self.wall_dir) if os.path.isfile(os.path.join(self.wall_dir, f))]
        length = len(files_log)
        return length

    def __getitem__(self, idx):
        """ Get data from disc"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # ======= Original =======
        # target_obj = get_raw_data(dir=self.target_obj_dir,
        #                           idx=idx)
        # disturb_obj = get_raw_data(dir=self.disturb_dir,
        #                            idx=idx)
        # wall = get_raw_data(dir=self.wall_dir,
        #                     idx=idx)
        # red_obj = get_raw_data(dir=self.red_dir,
        #                        idx=idx)
        # cheeze_obj = get_raw_data(dir=self.cheeze_dir,
        #                           idx=idx)
        # sample = {'target_obj': target_obj,
        #           'disturb_obj': disturb_obj,
        #           'wall': wall,
        #           'red_obj': red_obj,
        #           'cheeze_obj': cheeze_obj}

        # ======= YCB ======
        key_word_data = []
        sample = {}
        for i in range(len(self.key_words)):
            dir = os.path.join(self.base_dir, self.key_words[i])
            key_word_data = get_raw_data(dir=dir,
                                         idx=idx)
            sample[self.key_words[i]] = key_word_data
        return sample


class Siamese_dataset(Dataset):
    """ Construt Displacement data set for training purpose"""
    def __init__(self, base_dir,
                 key_words=[],
                 transform=None):
        self.base_dir = base_dir
        # ==== Original =====
        # self.disturb_dir = os.path.join(self.base_dir, 'disturb_img')
        # self.disturb_mask_dir = os.path.join(self.base_dir, 'disturb_obj_mask')
        #
        # self.target_obj_dir = os.path.join(self.base_dir, 'obj_img')
        # self.target_obj_maskdir = os.path.join(self.base_dir, 'target_obj_mask')
        #
        self.wall_dir = os.path.join(self.base_dir, 'wall')
        # self.wall_mask_dir = os.path.join(self.base_dir, 'wall_mask')
        #
        # self.red_dir = os.path.join(self.base_dir, 'obj_red_img')
        # self.red_mask_dir = os.path.join(self.base_dir, 'red_mask')
        #
        # self.cheeze_dir = os.path.join(self.base_dir, 'obj_cheeze_img')
        # self.cheeze_mask_dir = os.path.join(self.base_dir, 'cheeze_mask')

        # ==== YCB ====
        self.key_words = key_words
        # ==== Common ====
        self.transform = transform

    def __len__(self):
        files_log = [f for f in os.listdir(self.wall_dir) if os.path.isfile(os.path.join(self.wall_dir, f))]
        length = len(files_log)
        return length

    def __getitem__(self, idx):
        """ Get data from disc"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}
        for i in range(len(self.key_words)):
            img_dir = os.path.join(self.base_dir, self.key_words[i])
            mask_dir = os.path.join(self.base_dir, self.key_words[i] + '_mask')
            key_word_data = get_rgbd_data(array_dir=img_dir,
                                          mask_dir=mask_dir,
                                          idx=idx)
            sample[self.key_words[i]] = key_word_data
        return sample

