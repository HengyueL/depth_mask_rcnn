# import struct
import time
import os
import numpy as np
import shovel_grasp.utils as utils
from shovel_grasp.simulation import vrep


class Robot(object):
    def __init__(self, obj_root_dir, num_obj, workspace_limits):
        self.num_obj = num_obj
        self.workspace_limits = workspace_limits
        self.workspace_center = [(workspace_limits[0][0] + workspace_limits[0][1])/2,
                                 (workspace_limits[1][1] + workspace_limits[1][0])/2,
                                 0.002]
        self.home_pos = [-0.35,
                         -0.1,
                         0.4]  # Gripper home pos
        self.wall_home = np.asarray([workspace_limits[0][0] + 0.15,
                                     workspace_limits[1][1] - 0.15,
                                     -0.015])
        self.target_home = np.asarray([workspace_limits[0][1] - 0.15,
                                       workspace_limits[1][0] + 0.1,
                                       0.15])
        self.disturb_home = np.asarray(self.workspace_center)
        self.disturb_home[2] = 0.4
        self.plane_home = np.asarray(self.workspace_center)

        self.shovel_height = 0.06
        # ----- Action rot angle dictionary -------
        self.shovel_angle = {}
        self.shovel_angle['360'] = np.asarray([np.deg2rad(245.),
                                             np.deg2rad(360.),
                                             np.deg2rad(450.)])
        self.shovel_angle['0'] = np.asarray([np.deg2rad(-115.),
                                             np.deg2rad(0.),
                                             np.deg2rad(90.)])
        self.shovel_angle['5'] = np.asarray([np.deg2rad(360. - 115.),
                                             np.deg2rad(-4.5),
                                             np.deg2rad(87.9)])
        self.shovel_angle['10'] = np.asarray([np.deg2rad(360. - 115.3),
                                             np.deg2rad(-9.05),
                                             np.deg2rad(85.7)])
        self.shovel_angle['15'] = np.asarray([np.deg2rad(360.-116.),
                                              np.deg2rad(-13.6),
                                              np.deg2rad(83.5)])
        self.shovel_angle['20'] = np.asarray([np.deg2rad(360. - 116.4),
                                              np.deg2rad(-18.1),
                                             np.deg2rad(81.28)])
        self.shovel_angle['25'] = np.asarray([np.deg2rad(360. - 117.23),
                                              np.deg2rad(-22.52),
                                              np.deg2rad(78.85)])
        self.shovel_angle['30'] = np.asarray([np.deg2rad(360.-118.3),
                                              np.deg2rad(-27.),
                                              np.deg2rad(76.3)])
        self.shovel_angle['35'] = np.asarray([np.deg2rad(360. - 119.65),
                                              np.deg2rad(-31.32),
                                              np.deg2rad(73.52)])
        self.shovel_angle['40'] = np.asarray([np.deg2rad(360. - 121.33),
                                              np.deg2rad(-35.63),
                                              np.deg2rad(70.47)])
        self.shovel_angle['45'] = np.asarray([np.deg2rad(360.-123.5),
                                              np.deg2rad(-40.),
                                              np.deg2rad(67.)])
        self.shovel_angle['50'] = np.asarray([np.deg2rad(360. - 125.96),
                                              np.deg2rad(-43.97),
                                              np.deg2rad(63.27)])
        self.shovel_angle['55'] = np.asarray([np.deg2rad(360. - 129.11),
                                              np.deg2rad(-47.94),
                                              np.deg2rad(58.9)])
        self.shovel_angle['60'] = np.asarray([np.deg2rad(360.-133.0),
                                              np.deg2rad(-51.7),
                                              np.deg2rad(53.8)])
        self.shovel_angle['65'] = np.asarray([np.deg2rad(360. - 137.81),
                                              np.deg2rad(-55.22),
                                              np.deg2rad(47.81)])
        self.shovel_angle['70'] = np.asarray([np.deg2rad(360. - 143.74),
                                              np.deg2rad(-58.39),
                                              np.deg2rad(40.74)])
        self.shovel_angle['75'] = np.asarray([np.deg2rad(360.-151.),
                                              np.deg2rad(-61.),
                                              np.deg2rad(32.4)])
        self.shovel_angle['80'] = np.asarray([np.deg2rad(360. - 159.58),
                                              np.deg2rad(-63.2),
                                              np.deg2rad(22.65)])
        self.shovel_angle['85'] = np.asarray([np.deg2rad(360. - 169.41),
                                              np.deg2rad(-64.5),
                                              np.deg2rad(11.7)])
        self.shovel_angle['90'] = np.asarray([np.deg2rad(180.),
                                              np.deg2rad(-65.),
                                              np.deg2rad(0.)])
        self.shovel_angle['180'] = np.asarray([np.deg2rad(245.),
                                               np.deg2rad(360.),
                                               np.deg2rad(90.)])

        #  Check shovel_angle dic to see if it is correct

        self.grasp_ori = np.asarray([-np.pi, 0., np.pi/2])  # Grasp target Ori (for gripper rotation = 0)

        self.wall_handles = []   # A list of all wall handles
        self.target_handles = []
        self.plane_handles = []
        self.disturb_handles = []

        # Read files in object mesh directory
        self.obj_root_dir = obj_root_dir
        # ==== To Do ===
        self.target_obj_dir = os.path.join(self.obj_root_dir, 'obj')
        self.disturb_obj_dir = os.path.join(self.obj_root_dir, 'disturbs')
        self.wall_dir = os.path.join(self.obj_root_dir, 'walls')
        self.plane_dir = os.path.join(self.obj_root_dir, 'planes')
        self.ycb_dir = os.path.join(self.obj_root_dir, 'YCB')
        # This target_file_list is used only for siamese data collection
        self.target_file_list = [f for f in os.listdir(self.target_obj_dir) if (os.path.isfile(os.path.join(self.target_obj_dir, f)) and ('cheeze' in f))]
        # self.target_file_list = [f for f in os.listdir(self.target_obj_dir) if os.path.isfile(os.path.join(self.target_obj_dir, f))]
        self.disturb_file_list = [f for f in os.listdir(self.disturb_obj_dir) if os.path.isfile(os.path.join(self.disturb_obj_dir, f))]
        self.wall_file_list = [f for f in os.listdir(self.wall_dir) if os.path.isfile(os.path.join(self.wall_dir, f))]
        self.plane_file_list = [f for f in os.listdir(self.plane_dir) if os.path.isfile(os.path.join(self.plane_dir, f))]
        self.ycb_file_list = [f for f in os.listdir(self.ycb_dir) if os.path.isfile(os.path.join(self.ycb_dir, f))]
        self.ycb_file_list.sort()
        # Connect to simulator
        vrep.simxFinish(-1)  # Just in case, close all opened connections
        self.sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to V-REP on port 19997
        if self.sim_client == -1:
            print('Failed to connect to simulation (V-REP remote API server). Exiting.')
            exit()
        else:
            print('Connected to simulation.')
            self.stop_sim()
            self.restart_sim()
        # Setup virtual camera in simulation
        self.setup_sim_camera()
        self.open_RG2_gripper()

    def setup_sim_camera(self, resolution_x=640, resolution_y=480):
        # Get handle to camera
        perspectiveAngle = np.deg2rad(54.70)
        self.cam_intrinsics = np.asarray([[resolution_x / (2 * np.tan(perspectiveAngle / 2)), 0, resolution_x / 2],
                                          [0, resolution_y / (2 * np.tan(perspectiveAngle / 2)), resolution_y / 2],
                                          [0, 0, 1]])
        sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp',
                                                            vrep.simx_opmode_blocking)
        # Get camera pose and intrinsics in simulation
        sim_ret, cam_position = vrep.simxGetObjectPosition(self.sim_client, self.cam_handle, -1,
                                                           vrep.simx_opmode_blocking)
        sim_ret, cam_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.cam_handle, -1,
                                                                 vrep.simx_opmode_blocking)
        cam_trans = np.eye(4, 4)
        cam_trans[0:3, 3] = np.asarray(cam_position)
        # cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        cam_orientation = [cam_orientation[0], cam_orientation[1], cam_orientation[2]]
        cam_rotm = np.eye(4, 4)
        # cam_rotm[0:3,0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
        cam_rotm[0:3, 0:3] = utils.euler2rotm(cam_orientation)
        self.cam_pose = np.dot(cam_trans, cam_rotm)  # Compute rigid transformation representating camera pose
        self.cam_depth_scale = 1

    def add_target(self):
        self.add_object(add_obj_position=self.target_home,
                        pos_noise_level=0.08,
                        obj_type='target')
        # position = np.asarray(self.wall_home).copy()
        # position[1] -= 0.06
        # self.add_object(add_obj_position=self.workspace_center,
        #                 pos_noise_level=0.1,
        #                 obj_type='target'
        #                 )
        # return position

    def add_wall(self):
        orientation = np.asarray([0,
                                  0,
                                  2 * np.pi * np.random.random_sample()])
        self.add_object(add_obj_position=self.wall_home,
                        pos_noise_level=0.05,
                        obj_orientation=orientation,
                        obj_type='wall')
        # self.add_object(add_obj_position=self.workspace_center,
        #                 pos_noise_level=0.1,
        #                 obj_orientation=orientation,
        #                 obj_type='wall')

    def add_plane(self):
        # orientation = np.asarray([0., 0., 0.])
        # self.add_object(add_obj_position=self.plane_home,
        #                 pos_noise_level=-1,
        #                 obj_orientation=orientation,
        #                 obj_type='plane')
        pass

    def add_disturb(self):
        self.add_object(add_obj_position=self.disturb_home,
                        pos_noise_level=0.1,
                        obj_type='disturb')
        # self.add_object(add_obj_position=self.workspace_center,
        #                 pos_noise_level=0.1,
        #                 obj_type='disturb')

    def add_ycb(self, ycb_idx=0):
        position = np.asarray(self.workspace_center)
        position[2] = 0.25
        return self.add_object(add_obj_position=position,
                               pos_noise_level=0.05,
                               obj_type='ycb',
                               ycb_idx=ycb_idx)[:-4]

    def add_object(self, add_obj_position,
                   pos_noise_level=-1.,
                   obj_orientation=None,
                   obj_type=None,
                   ycb_idx=0):
        if obj_type == 'target':
            add_obj_list = self.target_handles
            obj_idx = np.random.randint(0, len(self.target_file_list))
            add_file = os.path.join(self.target_obj_dir, self.target_file_list[obj_idx])
        elif obj_type == 'disturb':
            add_obj_list = self.disturb_handles
            obj_idx = np.random.randint(0, len(self.disturb_file_list))
            add_file = os.path.join(self.disturb_obj_dir, self.disturb_file_list[obj_idx])
        elif obj_type == 'wall':
            add_obj_list = self.wall_handles
            obj_idx = np.random.randint(0, len(self.wall_file_list))
            add_file = os.path.join(self.wall_dir, self.wall_file_list[obj_idx])
        elif obj_type == 'plane':
            add_obj_list = self.plane_handles
            obj_idx = np.random.randint(0, len(self.plane_file_list))
            add_file = os.path.join(self.plane_dir, self.plane_file_list[obj_idx])
        elif obj_type == 'ycb':
            add_obj_list = self.disturb_handles
            obj_idx = ycb_idx
            file_name = self.ycb_file_list[obj_idx]
            add_file = os.path.join(self.ycb_dir, file_name)
        else:
            print('Invalid Obj Type')
            return
        _, obj_handle = vrep.simxLoadModel(self.sim_client, add_file,
                                           1, vrep.simx_opmode_blocking)
        if pos_noise_level > 0:
            pos_noise = pos_noise_level * np.random.random_sample((3,))
            pos_noise[2] = 0
        else:
            pos_noise = np.zeros([3])
        position = np.add(add_obj_position, pos_noise)
        if obj_orientation is None:
            obj_orientation = 2 * np.pi * np.random.random_sample((3,))
        self.set_single_obj_orientation(obj_handle, obj_orientation)
        self.set_single_obj_position(obj_handle, position)
        add_obj_list.append(obj_handle)
        time.sleep(0.5)
        if obj_type == 'ycb':
            return file_name

    def restart_sim(self):
        self.stop_sim()
        sim_ret, self.UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_target',
                                                                   vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, self.home_pos,
                                   vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client,
                                      self.UR5_target_handle, -1,
                                      self.shovel_angle['0'],
                                      vrep.simx_opmode_blocking)

        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(1)
        sim_ret, self.RG2_tip_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_tip', vrep.simx_opmode_blocking)
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1,
                                                               vrep.simx_opmode_blocking)
        while gripper_position[2] > self.home_pos[2] + 0.01:  # V-REP bug requiring multiple starts and stops to restart
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(1)
            sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1,
                                                                   vrep.simx_opmode_blocking)
        self.open_RG2_gripper()
        self.wall_handles = []
        self.plane_handles = []
        self.disturb_handles = []
        self.target_handles = []
        # self.add_plane()
        # self.add_wall()
        # self.add_disturb()
        self.add_target()

    def stop_sim(self):
        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(0.3)

    def check_sim(self):
        # Check if simulation is stable by checking if gripper is within workspace
        # Need to be modify, not working now
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        sim_ok = gripper_position[0] > self.workspace_limits[0][0] - 0.15 and gripper_position[0] < self.workspace_limits[0][1] + 0.15 and gripper_position[1] > self.workspace_limits[1][0] - 0.15 and gripper_position[1] < self.workspace_limits[1][1] + 0.15 and gripper_position[2] > 0. and gripper_position[2] < 0.5

        obj_ok = True
        if self.target_handles:
            obj_pos = self.get_single_obj_position(self.target_handles[0])
            if obj_pos[2] > 0.15:
                obj_ok = False
            if obj_pos[1] < self.workspace_limits[1][0] + 0.04 or obj_pos[1] > self.workspace_limits[1][1] - 0.04:
                obj_ok = False
            if obj_pos[0] < self.workspace_limits[0][0] + 0.04 or obj_pos[0] > self.workspace_limits[0][1] - 0.04:
                obj_ok = False

        if not sim_ok or not obj_ok:
            print('Simulation unstable. Restarting environment.')
            self.stop_sim()
            self.restart_sim()
        return sim_ok

    def get_single_obj_position(self, object_handle):
        _, obj_position = vrep.simxGetObjectPosition(self.sim_client,
                                                     object_handle,
                                                     -1,
                                                     vrep.simx_opmode_blocking)
        return np.asarray(obj_position)

    def get_single_obj_orientations(self, object_handle):
        _, obj_orientation = vrep.simxGetObjectOrientation(self.sim_client, object_handle, -1,
                                                           vrep.simx_opmode_blocking)
        a = []
        for i in obj_orientation:
            temp = i if i >= 0 else i + 2 * np.pi
            a.append(temp)
        return np.asarray(a)

    def set_single_obj_position(self, object_handle, goal_pos):
        _ = vrep.simxSetObjectPosition(self.sim_client,
                                       object_handle,
                                       -1,
                                       goal_pos,
                                       vrep.simx_opmode_blocking)

    def set_single_obj_orientation(self, object_handle, goal_ori):
        _ = vrep.simxSetObjectOrientation(self.sim_client,
                                          object_handle,
                                          -1,
                                          goal_ori,
                                          vrep.simx_opmode_blocking)

    def get_camera_data(self):
        """
        Return a tuple containing (RGB_img, Depth_img)
        """
        # Get color image from simulation
        sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle, 0,
                                                                       vrep.simx_opmode_blocking)
        color_img = np.asarray(raw_image)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float) / 255
        color_img[color_img < 0] += 1
        color_img *= 255
        color_img = np.fliplr(color_img)
        color_img = color_img.astype(np.uint8)

        # Get depth image from simulation
        sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.cam_handle,
                                                                                vrep.simx_opmode_blocking)
        depth_img = np.asarray(depth_buffer)
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = np.fliplr(depth_img)
        zNear = 0.01
        zFar = 10
        depth_img = depth_img * (zFar - zNear) + zNear

        return color_img, depth_img

    def close_RG2_gripper(self, default_vel=-0.3, motor_force=500):
        # RG2 gripper function
        gripper_motor_velocity = default_vel
        gripper_motor_force = motor_force

        sim_ret, gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint',
                                                           vrep.simx_opmode_blocking)
        sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, gripper_handle,
                                                                    vrep.simx_opmode_blocking)
        vrep.simxSetJointForce(self.sim_client, gripper_handle,
                               gripper_motor_force, vrep.simx_opmode_blocking)

        vrep.simxSetJointTargetVelocity(self.sim_client, gripper_handle,
                                        gripper_motor_velocity,
                                        vrep.simx_opmode_blocking)

        gripper_fully_closed = False
        close_gripper_count = 0
        while gripper_joint_position > -0.04:  # Block until gripper is fully closed
            sim_ret, new_gripper_joint_position = vrep.simxGetJointPosition(self.sim_client,
                                                                            gripper_handle,
                                                                            vrep.simx_opmode_blocking)
            close_gripper_count += 1
            if new_gripper_joint_position < gripper_joint_position:
                close_gripper_count = 0
                gripper_joint_position = new_gripper_joint_position
            if close_gripper_count > 1:
                return gripper_fully_closed
        gripper_fully_closed = True
        return gripper_fully_closed

    def open_RG2_gripper(self, default_vel=0.5, motor_force=100):
        # RG2 Gripper
        gripper_motor_velocity = default_vel
        gripper_motor_force = motor_force
        sim_ret, gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint',
                                                           vrep.simx_opmode_blocking)

        _, _ = vrep.simxGetJointPosition(self.sim_client, gripper_handle,
                                         vrep.simx_opmode_blocking)

        vrep.simxSetJointForce(self.sim_client, gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.sim_client, gripper_handle, gripper_motor_velocity,
                                        vrep.simx_opmode_blocking)

    def move_linear(self, tool_position, num_steps=10):
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                                                  vrep.simx_opmode_blocking)
        move_direction = np.asarray([tool_position[0] - UR5_target_position[0],
                                     tool_position[1] - UR5_target_position[1],
                                     tool_position[2] - UR5_target_position[2]])
        num_move_steps = num_steps
        move_step = move_direction / num_move_steps

        for step_iter in range(num_move_steps):
            vrep.simxSetObjectPosition(self.sim_client,
                                       self.UR5_target_handle,
                                       -1,
                                       (UR5_target_position[0] + move_step[0],
                                        UR5_target_position[1] + move_step[1],
                                        UR5_target_position[2] + move_step[2]),
                                       vrep.simx_opmode_blocking)
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                                                      vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                   (tool_position[0], tool_position[1], tool_position[2]), vrep.simx_opmode_blocking)
        time.sleep(0.05)

    # This function rotates the shovel action to be 90 deg (hard code, don't ask)
    def shovel_rot(self, mode='0'):
        """ mode='d'   Browse Dictionary to see the desired gripper angle """
        UR5_target_orientation = self.get_single_obj_orientations(self.UR5_target_handle)
        target_ori = self.shovel_angle[mode]
        ori_direction = np.asarray([target_ori[0] - UR5_target_orientation[0],
                                    target_ori[1] - UR5_target_orientation[1],
                                    target_ori[2] - UR5_target_orientation[2]])
        rot_angle = int(mode)
        if rot_angle != 0:
            if rot_angle < 6:
                num_move_step = 25
            elif rot_angle > 90:
                num_move_step = 25
            else:
                num_move_step = int(rot_angle / 5)
            ori_move_step = ori_direction/num_move_step
            for step_iter in range(num_move_step):
                vrep.simxSetObjectOrientation(self.sim_client,
                                              self.UR5_target_handle,
                                              -1,
                                              (UR5_target_orientation[0]+ori_move_step[0],
                                               UR5_target_orientation[1]+ori_move_step[1],
                                               UR5_target_orientation[2]+ori_move_step[2]),
                                              vrep.simx_opmode_blocking)

                sim_ret, UR5_target_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle,
                                                                                -1,
                                                                                vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client,
                                          self.UR5_target_handle,
                                          -1,
                                          (UR5_target_orientation[0] + ori_move_step[0],
                                           UR5_target_orientation[1] + ori_move_step[1],
                                           UR5_target_orientation[2] + ori_move_step[2]),
                                          vrep.simx_opmode_blocking)

    def shovel(self, position, rotation_angle):
        """ If prev_action == 'grasp', the gripper should first rotate to a shovel position"""
        if position[2] > 0.15:  # criterion for safety
            print('Shovel on the obstacle should be avoided! ')
            return False
        rot_key = str(int(rotation_angle))
        self.shovel_rot(mode=rot_key)
        shovel_direction = np.asarray([-0.07 * np.sin(np.deg2rad(rotation_angle)),
                                       0.07 * np.cos(np.deg2rad(rotation_angle)),
                                       0.0])
        print('Shovel Direction: %04f' % rotation_angle)

        position_shovel = np.asarray(position).copy()
        position_shovel[1] = position_shovel[1]
        position_shovel[1] = position_shovel[1]
        position_shovel[2] = self.shovel_height
        shovel_start_pos = position_shovel - shovel_direction
        shovel_location_margin = 0.25
        location_above_target = shovel_start_pos.copy()
        location_above_target[2] = location_above_target[2] + shovel_location_margin
        location_above_end = position_shovel.copy()
        location_above_end[2] = location_above_end[2] + shovel_location_margin
        print('Executing: shovel at (%f, %f, %f)' % (position_shovel[0], position_shovel[1], position_shovel[2]))

        # Sequential Actions
        self.move_linear(shovel_start_pos)
        self.move_linear(position_shovel)
        _ = self.close_RG2_gripper()
        time.sleep(0.5)
        self. move_linear(location_above_end)

        # re-set default pose
        self.move_linear(self.home_pos)
        time.sleep(0.4)

        # Check if grasp is successful
        gripper_full_closed = self.close_RG2_gripper()
        object_position = self.get_single_obj_position(self.target_handles[0])
        grasp_success = not gripper_full_closed and object_position[2] > 0.2
        if grasp_success:
            self.remove_object(object_handle=self.target_handles[0],
                               obj_handle_list=self.target_handles)
        self.open_RG2_gripper()
        # These reset values need to verify through v-rep simulation
        # There may be only one correspondence for one v-rep version
        if rot_key == '90':
            # rotate back to shovel home orientation
            self.shovel_rot(mode='360')
        elif rot_key == '0':
            pass
        else:
            self.shovel_rot(mode='180')
        return grasp_success

    def remove_object(self, object_handle, obj_handle_list):
        idx = obj_handle_list.index(object_handle)
        obj_handle_list.pop(idx)
        _ = vrep.simxRemoveModel(self.sim_client, object_handle, vrep.simx_opmode_blocking)
        time.sleep(0.3)
