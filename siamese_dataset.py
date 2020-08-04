import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2


def transform_img(img, is_wall=False):
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
    return target_img


class Raw_dataset(Dataset):
    """ Construt Displacement data set for training purpose"""
    def __init__(self, base_dir, transform=None):
        self.base_dir = base_dir

        self.disturb_dir = os.path.join(self.base_dir, 'disturb_img')
        # self.target_obj_dir = os.path.join(self.base_dir, 'obj_img')  # action_position[0]
        self.target_obj_dir = os.path.join(self.base_dir, 'obj_img')
        self.wall_dir = os.path.join(self.base_dir, 'wall_img')

        self.transform = transform

    def __len__(self):
        files_log = [f for f in os.listdir(self.wall_dir) if os.path.isfile(os.path.join(self.wall_dir, f))]
        length = len(files_log)
        return length

    def __getitem__(self, idx):
        """ Get data from disc"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        target_obj = np.load(self.target_obj_dir + '/%d.npy' % idx)
        target_obj = transform_img(target_obj)
        target_obj = np.transpose(target_obj, [2, 0, 1])

        disturb_obj = np.load(self.disturb_dir + '/%d.npy' % idx)
        disturb_obj = transform_img(disturb_obj)
        disturb_obj = np.transpose(disturb_obj, [2, 0, 1])

        wall = np.load(self.wall_dir + '/%d.npy' % idx)
        wall = transform_img(wall)
        wall = np.transpose(wall, [2, 0, 1])

        sample = {'target_obj': target_obj,
                  'disturb_obj': disturb_obj,
                  'wall': wall}
        return sample


class Siamese_dataset(Dataset):
    """ Construt Displacement data set for training purpose"""
    def __init__(self, base_dir, transform=None):
        self.base_dir = base_dir

        self.disturb_dir = os.path.join(self.base_dir, 'disturb_img')
        self.disturb_mask_dir = os.path.join(self.base_dir, 'disturb_obj_mask')

        self.target_obj_dir = os.path.join(self.base_dir, 'obj_img')
        self.target_obj_maskdir = os.path.join(self.base_dir, 'target_obj_mask')

        self.wall_dir = os.path.join(self.base_dir, 'wall_img')
        self.wall_mask_dir = os.path.join(self.base_dir, 'wall_mask')

        self.transform = transform

    def __len__(self):
        files_log = [f for f in os.listdir(self.wall_dir) if os.path.isfile(os.path.join(self.wall_dir, f))]
        length = len(files_log)
        return length

    def __getitem__(self, idx):
        """ Get data from disc"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        target_obj = np.load(self.target_obj_dir + '/%d.npy' % idx)
        target_mask = np.load(self.target_obj_maskdir + '/%d.npy' % idx)
        disturb_obj = np.load(self.disturb_dir + '/%d.npy' % idx)
        disturb_mask = np.load(self.disturb_mask_dir + '/%d.npy' % idx)
        wall = np.load(self.wall_dir + '/%d.npy' % idx)
        wall_mask = np.load(self.wall_mask_dir + '/%d.npy' % idx)

        for i in range(target_obj.shape[2]):
            target_obj[:, :, i] = np.multiply(target_obj[:, :, i], target_mask)
            disturb_obj[:, :, i] = np.multiply(disturb_obj[:, :, i], disturb_mask)
            wall[:, :, i] = np.multiply(wall[:, :, i], wall_mask)

        target_obj = transform_img(target_obj)
        target_obj = np.transpose(target_obj, [2, 0, 1])
        disturb_obj = transform_img(disturb_obj)
        disturb_obj = np.transpose(disturb_obj, [2, 0, 1])
        wall = transform_img(wall)
        wall = np.transpose(wall, [2, 0, 1])

        sample = {'target_obj': target_obj,
                  'disturb_obj': disturb_obj,
                  'wall': wall}
        return sample
