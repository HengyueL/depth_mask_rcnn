# Gray-Depth based Mask RCNN & Shovel-and-Grasp Project

major requirements:
  PyTorch version 1.5.0
  CoppeliaSim (Vrep) version 4.0.0

1) A Mask RCNN model trained by WISDOM dataset to achive object-agnostic segmentation

  WISDOM dataset: https://sites.google.com/view/wisdom-dataset/dataset_links

  To start training the mask rcnn model, run: 'Training Script: torch_rnn_train.py'
  To evaluate and visualize mask rcnn training result, run: 'testrcnn.py'

  The Gray-Depth mask rcnn Model path: maskrcnn_training/sd_model

  Network input data type: a list of image [image], where image has shape [C, H, W]; channel C=3 and is arranged as [Gray, Gray, Depth] in range (0, 1) (Please see torchdataset.py   for details )

2) Shovel and Grasp Project

  Run: 'Vrep':
    For training, open scene: 'shovel_grasp/simulation/simulation.ttt'
    For testing, open scene:'shovel_grasp/simulation/test_scenes/xxx.ttt' % (xxx is the scene numbers)
  
  The training and testing of KIDQN, DQN, DQN-init method：
    run: 'main_kidqn.py' with different param configurations.

    For training, set 'is_testing = False'

    For KIDQN method: set 'method = 'KIDQN' '; 
    For DQN-init method: set 'method = 'DQN_init' ';
    For DQN method: set 'method = 'DQN' ';  

