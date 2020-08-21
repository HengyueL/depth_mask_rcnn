import os
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2gray


def get_gdl_data(rgb_array,
                 depth_array):
    """
    load rgb and depth image and convert it into a gray-depth-laplacian numpy.array
    """
    # ==== Load and Construct RGB data ====
    gray_data = rgb2gray(rgb_array)
    x, y = gray_data.shape
    gray_data.shape = (x, y, 1)
    # ==== Load and Construct Depth data ====
    depth_data_normalized = normalize_depth_data(depth_array[:, :, 0])
    depth_data_laplacian = get_depth_laplacian(depth_data_normalized)

    # ==== Merge Data ====
    data_merged = np.concatenate((gray_data,
                                  depth_data_normalized,
                                  depth_data_laplacian), axis=2)
    return data_merged


def get_gdd_data(rgb_array,
                 depth_array):
    # ==== Load and Construct RGB data ====
    gray_data = rgb2gray(rgb_array)
    x, y = gray_data.shape
    gray_data.shape = (x, y, 1)
    # ==== Load and Construct Depth data ====
    depth_data_normalized = normalize_depth_data(depth_array[:, :, 0])
    # depth_data_laplacian = get_depth_laplacian(depth_data_normalized)

    # ==== Merge Data ====
    data_merged = np.concatenate((gray_data,
                                  depth_data_normalized,
                                  depth_data_normalized), axis=2)
    return data_merged


def normalize_rgb_data(rgb_array):
    normalized_array = rgb_array / 256
    return normalized_array


def normalize_depth_data(depth_array):
    img_max = np.amax(depth_array)
    normalized_array = depth_array / img_max
    x, y = normalized_array.shape
    normalized_array.shape = (x, y, 1)
    return normalized_array


def get_depth_laplacian(depth_array):
    """
    Input is a normalized depth img in np.array
    """
    depth_array = ndimage.laplace(depth_array)
    return depth_array


class SdMaskDataSet(object):
    def __init__(self, root, transforms, is_train=False):
        self.root = root
        self.is_train = is_train
        self.transforms = transforms
        self.color_imgs = list(sorted(os.listdir(os.path.join(root, "color_ims"))))
        self.depth_imgs = list(sorted(os.listdir(os.path.join(root, "depth_ims"))))
        if self.is_train:
            self.masks = list(sorted(os.listdir(os.path.join(root, "modal_segmasks"))))
        else:
            self.masks = list(sorted(os.listdir(os.path.join(root, "modal_segmasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        color_img_path = os.path.join(self.root, "color_ims", self.color_imgs[idx])
        depth_img_path = os.path.join(self.root, "depth_ims", self.depth_imgs[idx])
        if self.is_train:
            # mask_path = os.path.join(self.root, "segmasks_filled", self.masks[idx])
            mask_path = os.path.join(self.root, "modal_segmasks", self.masks[idx])
        else:
            mask_path = os.path.join(self.root, "modal_segmasks", self.masks[idx])

        depth_img = np.asarray(Image.open(depth_img_path).convert("RGB")).astype(np.float)
        color_img = np.asarray(Image.open(color_img_path).convert("L")).astype(np.float) / 255.
        img = get_gdl_data(color_img,
                           depth_img)
        # img = get_gdd_data(color_img,
        #                    depth_img)

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.depth_imgs)


# ====== Class Test Script ======
# import matplotlib.pyplot as plt
# from maskrcnn_training.sd_model import get_transform
#
# data_root_dir = '../datasets/low-res'
# test_dataset = SdMaskDataSet(data_root_dir,
#                              is_train=True,
#                              transforms=get_transform(train=True))
# a = test_dataset[1]
# print()
