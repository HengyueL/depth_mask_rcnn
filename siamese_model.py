# Old scripts to construct different CNN for siamese models
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torchvision


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def conv3x3(in_planes, out_planes, stride=1, dilation=1, bias=False):
    """3x3 convolution"""
    kernel_size = np.asarray((3, 3))

    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2

    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=full_padding,
        dilation=dilation,
        bias=bias,
    )


def initialize_weights(initializer):
    def initialize(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            initializer(m.weight)
            if m.bias is not None:
                torch.nn.init.constant(m.bias, 0)
    return initialize


def create_linear_network(input_dim, output_dim, hidden_units=[],
                          output_activation=None):
    model = []
    units = input_dim
    for next_units in hidden_units:
        model.append(nn.Linear(units, next_units))
        model.append(nn.ReLU())
        units = next_units

    model.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        model.append(nn.ReLU())

    return nn.Sequential(*model).apply(
        initialize_weights(nn.init.xavier_normal))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()

        self.stride = stride
        self.dilation = dilation
        self.downsample = downsample

        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        out = self.relu(out)
        return out


class ResModel(nn.Module):
    def __init__(self, in_channels):
        super(ResModel, self).__init__()

        # resnet50 = list(torchvision.models.resnet18(pretrained=True).children())[:-2]
        resnet50 = list(torchvision.models.resnet18(pretrained=True).children())[:-2]
        resnet50[0] = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        nn.init.kaiming_normal(resnet50[0].weight.data)
        self.resnet = nn.Sequential(*resnet50)
        self.conv2d = conv3x3(in_planes=512, out_planes=1, stride=3)
        self.linear1 = nn.Linear(121, 20)
        self.linear2 = nn.Linear(20, 10)

    def forward(self, x):
        # features = self.resnet(x)
        out = self.resnet(x)
        out = self.conv2d(out)
        out = torch.relu(out.view(-1, 121))
        out = self.linear1(out)
        out = torch.tanh(self.linear2(out))
        return out


# class CnnNet(nn.Module):
#     def __init__(self, in_channels):
#         super(CnnNet, self).__init__()
#         self.cnn1 = nn.Sequential(
#             nn.Conv2d(in_channels, 64, kernel_size=10),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(2),
#
#             # nn.ReflectionPad2d(1),
#             nn.Conv2d(64, 128, kernel_size=8),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(128),
#             nn.MaxPool2d(2),
#
#             # nn.ReflectionPad2d(1),
#             nn.Conv2d(128, 128, kernel_size=4),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(128),
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(128, 256, kernel_size=4),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(256),
#         )
#
#         self.fc1 = nn.Sequential(
#             nn.Linear(256 * 12 * 12, 2056),
#             nn.Sigmoid(),
#         )
#
#         self.fc2 = nn.Sequential(
#             nn.Linear(2056, 1),
#             nn.Sigmoid(),
#         )
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward_once(self, x):
#         output = self.cnn1(x)
#         # print(output.size())
#         output = output.view(output.size()[0], -1)
#         # print(output.size())
#         output = self.fc1(output)
#         return output
#
#     def forward(self, input1, input2):
#         output1 = self.forward_once(input1)
#         output2 = self.forward_once(input2)
#         dis = torch.abs(output1 - output2)
#         out = self.fc2(dis)
#         return out


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin=0.95):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.sum()


# ------ Test Output Dimension ----------
# input_shape = (3, 500, 500)
# a = ResModel(in_channels=3).cuda()
# bs = 1
# input_1 = torch.rand(bs, *input_shape).cuda()
# output_feat = a(input_1).cpu()
# n_size = output_feat.data.size()
# # torch.cuda.m
# print(n_size)
# pass
