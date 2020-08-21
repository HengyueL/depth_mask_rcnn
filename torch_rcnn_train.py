from maskrcnn_training.engine import train_one_epoch, evaluate
import torch
import os
from maskrcnn_training.torchdataset import SdMaskDataSet
from maskrcnn_training.sd_model import get_transform, get_model_instance_segmentation
from maskrcnn_training import utils
import numpy as np
from skimage.color import rgb2gray
from scipy import ndimage
import matplotlib.pyplot as plt
from testrcnn import instance_segmentation_api
from PIL import Image


save_model_dir = 'save_file_dir/pytorch_gdl_test'
# save_model_dir = 'save_file_dir/pytorch_gdd_test'
if not os.path.exists(save_model_dir):
    os.mkdir(save_model_dir)


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    # dataset = SdMaskDataSet('datasets/sim_img',
    #                         is_train=True,
    #                         transforms=get_transform(train=True))
    dataset = SdMaskDataSet('datasets/low-res',
                            is_train=True,
                            transforms=get_transform(train=True))
    dataset_test = SdMaskDataSet('datasets/low-res',
                                 is_train=False,
                                 transforms=get_transform(train=False))
    # dataset_test = SdMaskDataSet('datasets/sim_img', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    # dataset = torch.utils.data.Subset(dataset, indices[0:350])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes,
                                            pretrained=True)

    # move model to the right device
    model.to(device, dtype=torch.float)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=5e-3,
                                momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.SGD(params, lr=1e-3,
    #                             momentum=0.9)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 50

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        torch.save(model.state_dict(),
                   os.path.join(save_model_dir, '%d.pth' % epoch))

    print("That's it!")


def test_eval():
    depth_dir = 'datasets/low-res/depth_ims'
    rgb_dir = 'datasets/low-res/color_ims'

    # # img_file = os.path.join(root_dir, 'image_000000.png')
    rgb_img_file = os.path.join(rgb_dir, 'image_000306.png')
    depth_img_file = os.path.join(depth_dir, 'image_000306.png')
    rgb_array = np.asarray(Image.open(rgb_img_file).convert("RGB"))
    depth_array = np.asarray(Image.open(depth_img_file).convert("RGB"))[:,:,0]

    # depth_array = np.load('depth.npy')
    # rgb_array = np.load('rgb.npy')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_path = os.path.join(save_model_dir,
                              '%d.pth' % 29)
    # ==== Normalize depth
    depth_array = depth_array / np.amax(depth_array)
    # depth_array = np.pad(depth_array,
    #                      pad_width=[100, 100])
    x, y = depth_array.shape
    depth_array.shape = (x, y, 1)
    # ==== Lap
    depth_lap = ndimage.laplace(depth_array)
    # ==== RGB to Gray
    gray_data = rgb2gray(rgb_array)
    # gray_data = np.pad(gray_data,
    #                    pad_width=[100, 100])
    gray_data.shape = (x, y, 1)
    # ==== Construch Input Data
    # input_data = np.concatenate((gray_data,
    #                              depth_array,
    #                              depth_array), axis=2)
    input_data = np.concatenate((gray_data,
                                 depth_array,
                                 depth_lap), axis=2)
    input_data = np.transpose(input_data, [2, 0, 1])
    input_data = torch.from_numpy(input_data).to(device=device,
                                                 dtype=torch.float)
    # get the model using our helper function
    model = get_model_instance_segmentation(2,
                                            pretrained=True)
    # move model to the right device
    model.to(device, dtype=torch.float)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # ====
    rcnn_output = model([input_data])[0]
    mask_set = rcnn_output['masks'].cpu().data.detach().numpy()
    box_set = rcnn_output['boxes'].cpu().data.detach().numpy()
    label_set = rcnn_output['labels'].cpu().data.detach().numpy()
    # score_set = rcnn_output['scores'].cpu().data.detach().numpy()

    # ==== Visualize Individual Masks
    # num_subplot = mask_set.shape[0]
    # fig = plt.figure(0)
    # for i in range(num_subplot):
    #     ax = fig.add_subplot(1, num_subplot+1, i+1)
    #     ax.imshow(mask_set[i, 0, :, :])
    # ax = fig.add_subplot(1, num_subplot+1, i+2)
    # ax.imshow(rgb_array)
    img_visual = (rgb_array / 5).astype(np.uint8)
    fig_1 = plt.figure(0)
    ax_1 = fig_1.add_subplot(1, 2, 1)
    ax_1.imshow(rgb_array)
    ax_2 = fig_1.add_subplot(1, 2, 2)
    instance_segmentation_api(img_visual, mask_set, box_set, label_set, fig=ax_2)
    plt.show()
    pass


if __name__ == "__main__":
    # parse the provided configuration file, set tf settings, and benchmark
    # test_eval()
    main()
