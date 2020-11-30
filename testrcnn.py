# This Script Contains the utility functions for the mask-rcnn result visualization
import cv2
import numpy as np
import random


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
