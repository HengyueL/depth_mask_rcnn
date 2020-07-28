# depth_mask_rcnn
Self implemented depth_mask

Training Script: torch_rnn_train.py

Gray-depth mask rcnn Model: maskrcnn_training/sd_model

Network input data type: a list of image [image], where image has shape [C, H, W]; channel C=3 and is arranged as [Gray, Gray, Depth] in range (0, 1) (Please see torchdataset.py for details )

