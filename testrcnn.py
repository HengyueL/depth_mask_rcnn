import torch, torchvision
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from depth_mask.engine import train_one_epoch, evaluate
from depth_mask.torchdataset import SdMaskDataSet
from depth_mask.sd_model import get_transform, get_model_instance_segmentation
import depth_mask.utils
from PIL import Image

def random_colour_masks(image):
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def instance_segmentation_api(img, masks, boxes, pred_cls, fig,
                              threshold=0.5, rect_th=3, text_size=3, text_th=3):
    for i in range(len(masks)):
        rgb_mask = np.where(masks[i, 0, :, :] > threshold, 1., 0.)
        rgb_mask = random_colour_masks(rgb_mask)
        if pred_cls[i] > 0:
            img = cv2.addWeighted(img, 1, rgb_mask, 0.25, 0)
            # box_start = (int(boxes[i][0]), int(boxes[i][1]))
            # box_end = (int(boxes[i][2]), int(boxes[i][3]))
            # if box_end[0]-box_start[0] < 280 and box_end[1]-box_start[1] < 280:
            #     cv2.rectangle(img, box_start, box_end , color=(0, 255, 0), thickness=rect_th)
                # cv2.putText(img,pred_cls[i], int(boxes[i][0]), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
                # plt.figure(figsize=(20,30))
    fig.imshow(img)

root_dir = 'datasets/low-res/depth_ims'
rgb_dir = 'datasets/low-res/color_ims'
save_model_dir = 'pytorch_sdrcnn_cocoeval'
num_classes = 2

# # img_file = os.path.join(root_dir, 'image_000000.png')
# rgb_img_file = os.path.join(rgb_dir, 'image_000306.png')
# depth_img_file = os.path.join(rgb_dir, 'image_000306.png')
#
# rgb_raw_img = np.asarray(Image.open(rgb_img_file).convert("RGB"))
# depth_raw_img = np.asarray(Image.open(depth_img_file).convert("RGB"))[:,:,0]
# ====
rgb_load = np.load('color.npy')
rgb_raw_img = np.zeros_like(rgb_load)
for i in range(rgb_load.shape[2]):
    rgb_raw_img[:, :, i] = rgb_load[:, :, 2-i]

depth_raw_img = np.load('depth.npy')[:, :, 0]

color_img = rgb_raw_img.astype(np.float) / 255.
img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]
for c in range(color_img.shape[2]):
    color_img[:, :, c] = (color_img[:, :, c] - img_mean[c]) / img_std[c]

depth_img = depth_raw_img.astype(np.float)
x, y = depth_img.shape
depth_img.shape = (x, y, 1)
img_max = np.amax(depth_img)
depth_img = depth_img / img_max

img = np.concatenate((color_img[:, :, 0:2], depth_img),
                     axis=2)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

test_input = [torch.from_numpy(np.transpose(img, [2, 0, 1])).to(device, dtype=torch.float)]
model = get_model_instance_segmentation(num_classes).to(device, dtype=torch.float)
model.load_state_dict(torch.load(os.path.join(save_model_dir, '19.pth')))
model.eval()

output = model(test_input)
box_list = output[0]['boxes'].cpu().detach().numpy()
mask_list = output[0]['masks'].cpu().detach().numpy()
cls_list = output[0]['labels'].cpu().detach().numpy()

img_visual = (rgb_raw_img / 5).astype(np.uint8)
fig_1 = plt.figure(0)
ax_1 = fig_1.add_subplot(1, 2, 1)
ax_1.imshow(rgb_raw_img)
ax_2 = fig_1.add_subplot(1, 2, 2)
instance_segmentation_api(img_visual, mask_list, box_list, cls_list, fig=ax_2)
plt.show()
print()
