# Gray-Depth based Mask RCNN & Shovel-and-Grasp Project

This repository include three part:
  1) Gray-depth-depth mask rcnn for object-agnostic segmentation for table top manipulations
  2) Siamese network to identify target object
  3) Shovel-and-Grasp experiment.

# Dependencies:
    General requirement:
        1) Numpy; PyTorch version 1.5.0
        2) CoppeliaSim (Vrep) version 4.0.0 
        3) scipy
        4) matplotlib
    Extra dependencies for mask rcnn training and eval:
        1) cython, pycocotools. See: https://github.com/pytorch/vision/tree/master/references/detection

# Running Tutorial
## Mask R-CNN
  1)  Download WISDOM dataset: https://sites.google.com/view/wisdom-dataset/dataset_links
      We will need the 'Real/low-res' one.
  2) Put the dataset under folder: 'SaG_and_maskRCNN/datasets/low-res/(**low_res dataset components**)'
  3) To start training the mask rcnn model, run: 'Training Script: torch_rnn_train.py'
  4) To evaluate and visualize mask rcnn training result, run: 'testrcnn.py'

## Siamese Network (Updated but not tested)
  1) Run Vrep, open scene 'simulation/simulation.ttt'
  2) Go to 'shovel_grasp/Robot.py', comment the lines:
        self.add_plane()
        self.add_wall()
        self.add_disturb()
        self.add_target()
      under function 'restart_sim(self)'
  3) Modify the drop object positions under function: add_target(self); add_disturb(self); add_wall(self), so that the object will not be dropped outside the heightmap of the        workspace.
  4) Modify the 'siamese_dataset.py' accordingly to construct dataset from the collected data in 3);
  5) Modify 'test_siamese.py' accordingly to train and test the siamese network. (This script is not updated, cannot run directly.)
  In order to upgrade the old version siamese into more general term that is able to detect more target objects, 4) and 5) must be updated and the siamese network must be retrained.
         
## Shovel and Grasp Project
  1)Run: 'Vrep':
      For training, open scene: 'shovel_grasp/simulation/simulation.ttt'
      For testing, open scene:'shovel_grasp/simulation/test_scenes/xxx.ttt' % (xxx is the scene numbers)

  2) The training and testing of KIDQN, DQN, DQN-init method：
      run: 'main_kidqn.py' with different param configurations discribed as follows:
      
        For training, set 'is_testing = False'
        
        For KIDQN method: set 'method = 'KIDQN' '; 
        For DQN-init method: set 'method = 'DQN_init' ';
        For DQN method: set 'method = 'DQN' ';  
    
  3) The Old Version Pretrained models that has been used in main_kidqn can be found here：
    
    https://drive.google.com/drive/folders/1ZwVBn2oMX2K-GyxsHjzFBVELFxk7H3Z7?usp=sharing
    
      1) Pretrained maskrcnn should bu put under 'save_model_dir/pytorch_gdd_test/29.pth'
  
      2) Old siamese models to detect one single target object (with workspace heightmap input) should locate under the path 'save_model_dir/Siamese_recollect/siamese-99.pth'
    
      3) The visual affordance network model should be put under path specified by variable 'model_logger_dir' in the main function.
  
  4) For future work of using a Siamese Network (using image patch as input, not the entire workspace heightmap size), the siamese network related code in 'main_kidqn' and            'siamese_model.py' must be modified accordingly.
    
