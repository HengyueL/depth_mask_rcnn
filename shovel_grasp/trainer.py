import os
import numpy as np
import cv2
import torch
from scipy import ndimage
from shovel_grasp.model import RotModelRes


class TrainerDQN(object):
    def __init__(self,
                 action_space_size,
                 future_reward_discount,
                 save_model_dir=None):

        self.action_space_size = action_space_size
        self.iteration = 0
        self.num_of_rotation = 3  # 0, 45, 90 3-angle motion primitives
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.shovel_model = RotModelRes(self.device).to(self.device)
        self.target_model = RotModelRes(self.device).to(self.device)

        # Load pre-trained model
        if save_model_dir is not None:
            pre_trained_model = os.path.join(save_model_dir,
                                             '1160_shovel_DQN_model.pth')
            if os.path.exists(pre_trained_model):
                #  Change the path here if the continue training
                self.shovel_model.load_state_dict(torch.load(pre_trained_model))
                print('Pre-trained model loaded')

        self.future_reward_discount = future_reward_discount
        self.criterion = torch.nn.SmoothL1Loss().to(self.device)  # Huber loss

        # Set model to training mode
        self.shovel_model.train()
        self.target_model.load_state_dict(self.shovel_model.state_dict())

        # Initialize optimizer
        self.shovel_optimizer = torch.optim.SGD([{'params': self.shovel_model.feature_trunk.parameters(), 'lr': 1e-5},
                                                 {'params': self.shovel_model.q_func_cnn.parameters()}
                                                 ],
                                                lr=1e-5, momentum=0.9,
                                                weight_decay=2e-5)

        # Initialize lists to save execution info and RL variables
        self.reward_value_log = []
        self.training_loss_log = []

    def make_predictions(self, input_img,
                         output_size,
                         requires_grad=False,
                         mode='predict'):
        """

        :param input_img: shape (N, N, 4) --- image standard coordinate (x, y)
        :param requires_grad: default false --- this prediction does not contribute to backprop
        :param mode:
        :return:
        """
        shovel_q_value = []
        for i in range(self.num_of_rotation):
            # rot_idx = i
            input_data = np.asarray([input_img], dtype=float)
            # Formulate Input Tensor (Variable, Volatile = True)
            input_data = torch.from_numpy(input_data).permute(0, 3, 1, 2).to(device=self.device, dtype=torch.float)
            input_data.requires_grad = requires_grad

            # Feed Forward
            if mode == 'predict':
                shovel_out = self.shovel_model.forward(input_data, rot_idx=i)
            else:
                shovel_out = self.target_model.forward(input_data, rot_idx=i)

            shovel_prediction = shovel_out[0][0].cpu().data.detach().numpy()
            pad_start = int((shovel_prediction.shape[0] - output_size) / 2)
            shovel_q_value.append(shovel_prediction[pad_start:pad_start+output_size,
                                                    pad_start:pad_start+output_size])
        return np.asarray(shovel_q_value)

    def get_label_value(self,
                        shovel_success,
                        next_img_input):
        # Compute current reward
        if shovel_success:
            current_reward = 1.0
        else:
            current_reward = 0.0
        next_shovel_predictions = self.make_predictions(next_img_input,
                                                        output_size=self.action_space_size,
                                                        requires_grad=False,
                                                        mode='target')
        future_reward = np.amax(next_shovel_predictions)
        expected_reward = current_reward + self.future_reward_discount * future_reward
        print('Expected reward: %f + %f x %f = %f' % (current_reward,
                                                      self.future_reward_discount,
                                                      future_reward,
                                                      expected_reward))
        return expected_reward, current_reward

    def backprop_mask(self, input_img,
                      rot_idx,
                      action_position,
                      output_size,
                      label_value,
                      prev_mask):
        self.shovel_optimizer.zero_grad()
        input_data = np.asarray([input_img], dtype=np.float)
        input_data = torch.from_numpy(input_data).permute(0, 3, 1, 2).to(self.device, dtype=torch.float)
        input_data.requires_grad = True
        shovel_predictions = self.shovel_model.forward(input_data,
                                                       rot_idx=rot_idx)

        label_numpy = shovel_predictions.clone().cpu().data.detach().numpy()
        # -------
        weight = 1e-1
        label_weights = np.ones_like(label_numpy) * weight
        # -------
        pad_width = int((label_numpy.shape[2] - output_size) / 2)
        label_numpy[0, 0,
        pad_width:pad_width + output_size,
        pad_width:pad_width + output_size] = np.multiply(label_numpy[0, 0,
                                                         pad_width:pad_width + output_size,
                                                         pad_width:pad_width + output_size],
                                                         prev_mask)
        label_numpy[0, 0,
                    action_position[0] + pad_width,
                    action_position[1] + pad_width] = label_value
        label_weights[0, 0,
                      action_position[0] + pad_width,
                      action_position[1] + pad_width] = 1.

        label = torch.from_numpy(label_numpy).to(self.device, dtype=torch.float)
        label.requires_grad = True
        label_weights = torch.from_numpy(label_weights).to(self.device, dtype=torch.float)
        label_weights.requires_grad = False

        loss = self.criterion(shovel_predictions, label) * label_weights
        loss = loss.sum()
        loss.backward()
        loss_value = loss.cpu().data.detach().numpy()
        self.shovel_optimizer.step()
        print('Training Loss: %f' % loss_value)
        return loss_value

    def update_target_net(self):
        self.target_model.load_state_dict(self.shovel_model.state_dict())
        print('Target Model Updated')


def get_prediction_vis(predictions, color_heightmap, action_rot, action_position, num_rotation=3):
    canvas = None
    tmp_row_canvas = None
    zoom_scale = float(color_heightmap.shape[0])/predictions.shape[1]
    for canvas_col in range(num_rotation):
        rotate_idx = canvas_col
        prediction_vis = predictions[rotate_idx,:,:].copy()
        prediction_vis = ndimage.zoom(prediction_vis, zoom=[zoom_scale, zoom_scale], mode='nearest')
        prediction_vis = np.clip(prediction_vis, 0, 5)
        prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
        prediction_vis = (0.5*cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2BGR) + 0.5*prediction_vis).astype(np.uint8)
        if rotate_idx == action_rot:
            cv2.circle(prediction_vis,
                       (int(action_position[1] * zoom_scale - int(zoom_scale/2)),
                        int(action_position[0] * zoom_scale - int(zoom_scale/2))),
                       15, (255, 0, 0), 8)
        prediction_vis = ndimage.rotate(prediction_vis, rotate_idx * 45., reshape=False)
        prediction_vis = cv2.flip(prediction_vis, 0)
        if tmp_row_canvas is None:
            tmp_row_canvas = prediction_vis
        else:
            tmp_row_canvas = np.concatenate((tmp_row_canvas,prediction_vis), axis=1)
    if canvas is None:
        canvas = tmp_row_canvas
    else:
        canvas = np.concatenate((canvas,tmp_row_canvas), axis=0)
    return canvas
