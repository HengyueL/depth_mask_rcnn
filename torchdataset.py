import os
import numpy as np
import torch
from PIL import Image


class SdMaskDataSet(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.color_imgs = list(sorted(os.listdir(os.path.join(root, "color_ims"))))
        self.depth_imgs = list(sorted(os.listdir(os.path.join(root, "depth_ims"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "modal_segmasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        color_img_path = os.path.join(self.root, "color_ims", self.color_imgs[idx])
        depth_img_path = os.path.join(self.root, "depth_ims", self.depth_imgs[idx])
        mask_path = os.path.join(self.root, "modal_segmasks", self.masks[idx])

        depth_img = np.asarray(Image.open(depth_img_path).convert("RGB")).astype(np.float)
        depth_img = depth_img[:, :, 0]
        x, y = depth_img.shape
        depth_img.shape = (x, y, 1)
        img_max = np.amax(depth_img)
        depth_img = depth_img / img_max

        color_img = np.asarray(Image.open(color_img_path).convert("L")).astype(np.float) / 255.
        x, y = color_img.shape
        color_img.shape = (x, y, 1)
        img = np.concatenate((color_img, color_img, depth_img),
                             axis=2)
        # color_img = np.asarray(Image.open(color_img_path).convert("RGB")).astype(np.float) / 255.
        # img_mean = [0.485, 0.456, 0.406]
        # img_std = [0.229, 0.224, 0.225]
        # for c in range(color_img.shape[2]):
        #     color_img[:, :, c] = (color_img[:, :, c] - img_mean[c]) / img_std[c]
        # img = np.concatenate((color_img[:, :, 0:2], depth_img),
        #                      axis=2)

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