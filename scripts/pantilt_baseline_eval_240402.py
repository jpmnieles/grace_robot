import os
import sys
sys.path.append(os.getcwd())

import math
import time
import yaml
import pandas as pd
import pickle
from datetime import datetime

import rospy
import message_filters
from hr_msgs.msg import TargetPosture, MotorStateList
from rospy_message_converter import message_converter
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, String, Header
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped

import tf
from tf.transformations import translation_matrix, quaternion_matrix
from sensor_msgs.msg import JointState

import cv2
import dlib
import numpy as np
import random
import threading
import copy

from grace.utils import *
from aec.baseline import BaselineCalibration
from grace.attention import *


class VisuoMotorNode(object):

    rl_state = {  # RL environment on the network takes care of the other side
        'chess_idx': None,
        'theta_left_pan': None,
        'theta_right_pan': None,
        'theta_tilt': None,
        'chest_cam_px_x_tminus1': None,
        'chest_cam_px_y_tminus1': None,  
        'left_eye_px_x_tminus1': None,
        'left_eye_px_y_tminus1': None,
        'right_eye_px_x_tminus1': None,
        'right_eye_px_y_tminus1': None,
        'chest_cam_px_x': None,
        'chest_cam_px_y': None,  
        'left_eye_px_x': None,
        'left_eye_px_y': None,
        'right_eye_px_x': None,
        'right_eye_px_y': None,
        'dx_l': None,
        'dy_l': None,
        'dx_r': None,
        'dy_r': None,
        '3d_point': None,
        'chest_pan_angle': None,
        'chest_tilt_angle': None,
        'plan_phi_left_pan': None,
        'plan_phi_right_pan': None,
        'plan_phi_tilt': None,
        'chest_img': None,
        'left_eye_img': None,
        'right_eye_img': None,
        'depth_img': None,
        'chest_img_stamp': None,
        'left_eye_img_stamp': None,
        'right_eye_img_stamp': None,
        'depth_img_stamp': None,
        'theta_left_pan_cmd': None,
        'theta_right_pan_cmd': None,
        'theta_tilt_cmd': None,
    }

    state_list = {
        'chess_idx': [],
        'theta_left_pan': [],
        'theta_right_pan': [],
        'theta_tilt': [],
        'chest_cam_px_x_tminus1': [],
        'chest_cam_px_y_tminus1': [], 
        'left_eye_px_x_tminus1': [],
        'left_eye_px_y_tminus1': [],
        'right_eye_px_x_tminus1': [],
        'right_eye_px_y_tminus1': [],
        'chest_cam_px_x': [],
        'chest_cam_px_y': [],  
        'left_eye_px_x': [],
        'left_eye_px_y': [],
        'right_eye_px_x': [],
        'right_eye_px_y': [],
        'dx_l': [],
        'dy_l': [],
        'dx_r': [],
        'dy_r': [],
        '3d_point': [],
        'chest_pan_angle': [],
        'chest_tilt_angle': [],
        'plan_phi_left_pan': [],
        'plan_phi_right_pan': [],
        'plan_phi_tilt': [],
        'chest_img_stamp': [],
        'left_eye_img_stamp': [],
        'right_eye_img_stamp': [],
        'depth_img_stamp': [],
        'theta_left_pan_cmd': [],
        'theta_right_pan_cmd': [],
        'theta_tilt_cmd': [],
    }

    joints_list = ['neck_roll', 'neck_pitch', 'neck_yaw',
                   'head_roll', 'head_pitch', 
                   'eyes_pitch', 'lefteye_yaw', 'righteye_yaw']
    
    pickle_data = {
        'subject_num': None,
        'markers': [-20, -10, 0, 10, 20],
        'trial_num': None,
        'data': [],
    }
    marker_list = [-20, -10, 0, 10, 20]

    def __init__(self, num_trials, motors=["EyeTurnLeft", "EyeTurnRight", "EyesUpDown"], degrees=True):
        self.num_trials = num_trials 
        self.start = time.time()
        self.num_ctr = num_trials*2
        self.ctr_tilt = 0
        self.motor_lock = threading.Lock()
        self.buffer_lock = threading.Lock()
        self.action_lock = threading.Lock()
        self.motors = motors
        self.degrees = degrees
        self._set_motor_limits(motors)

        self.attention = ChessboardAttention()
        self.calibration = BaselineCalibration(self.buffer_lock)  # RL Model
        self.calibration.toggle_backlash(True)
        self.frame_stamp_tminus1 = rospy.Time.now()
        self.motor_stamp_tminus1 = rospy.Time.now()
        self.chess_idx_tminus1 = 0

        self.motor_pub = rospy.Publisher('/hr/actuators/pose', TargetPosture, queue_size=1)
        self.camera_mtx = load_json("config/camera/camera_mtx.json")
        self.camera_mtx_params = load_camera_mtx()
        self.bridge = CvBridge()
        time.sleep(1)
        
        # Initial Reset Action
        self.move((-18, -18, 22))
        time.sleep(0.5)
        self.move((0, 0, 0))
        time.sleep(1.0)

        self.action = None

        self.motor_sub = rospy.Subscriber('/hr/actuators/motor_states', MotorStateList, self._capture_state)
        self.left_eye_sub = message_filters.Subscriber("/left_eye/image_raw", Image)
        self.right_eye_sub = message_filters.Subscriber("/right_eye/image_raw", Image)  # TODO: change to right eye when there is better camera
        self.chest_cam_sub = message_filters.Subscriber('/hr/perception/jetson/realsense/camera/color/image_raw', Image)
        self.depth_cam_sub = message_filters.Subscriber('/hr/perception/jetson/realsense/camera/aligned_depth_to_color/image_raw', Image)
        self.ats = message_filters.ApproximateTimeSynchronizer([self.left_eye_sub, self.right_eye_sub, 
                                                                self.chest_cam_sub, self.depth_cam_sub], queue_size=1, slop=0.25)
        self.ats.registerCallback(self.eye_imgs_callback)
        self.rt_display_pub = rospy.Publisher('/output_display1', Image, queue_size=1)
        # self.point_pub = rospy.Publisher('/point_location', PointStamped, queue_size=1)
        # self.tf_listener = tf.TransformListener()

        self.chess_idx = 0
        self.ctr = 0
        self.disp_img = np.zeros((480,640,3), dtype=np.uint8)
        self.calib_params = load_json('config/calib/calib_params.json')
        rospy.loginfo('Running')

    def set_action(self, action):
        with self.action_lock:
            self.action = action

    def _capture_state(self, msg):
        """Callback for capturing motor state
        """
        eye_motors_list = []
        self._msg = msg
        for idx, x in enumerate(msg.motor_states):
            if x.name in self.names:
                eye_motors_list.append(idx)
        if len(eye_motors_list) == 3:
            # rospy.loginfo('Complete')
            # rospy.loginfo(msg.motor_states[eye_motors_list[0]])
            # rospy.loginfo(msg.motor_states[eye_motors_list[1]])
            # rospy.loginfo(msg.motor_states[eye_motors_list[2]])
            with self.motor_lock:
                for i in range(self.num_names):
                    motor_msg = msg.motor_states[eye_motors_list[i]]
                    idx = self.motors.index(motor_msg.name)
                    self._motor_states[idx] = message_converter.convert_ros_message_to_dictionary(motor_msg)
                    self._motor_states[idx]['angle'] = motor_int_to_angle(motor_msg.name, motor_msg.position, self.degrees)
        else:
            # rospy.loginfo('Incomplete')
            pass

        
    def depth_to_pointcloud(self, px, depth_img, camera_mtx, z_replace=1.0):  
        fx = camera_mtx[0][0]
        cx = camera_mtx[0][2]
        fy = camera_mtx[1][1]
        cy = camera_mtx[1][2]
        u = round(px[0])
        v = round(px[1])
        z = depth_img[v,u]/1000.0
        z = z - 0.0042  # Added realsense discrepancy
        if z<=0:
            z = z_replace
        x = ((u-cx)/fx)*z
        y = ((v-cy)/fy)*z
        return x,y,z

    def transform_points(self, pts, T_mtx):
        new_obj_pts = []
        for pt in pts:
            temp_pt = np.append(pt, 1).reshape(-1,1)
            temp_pt2 = (T_mtx @ temp_pt).squeeze()
            new_obj_pts.append(temp_pt2[:3])
        new_obj_pts = np.array(new_obj_pts)
        return new_obj_pts


    def eye_imgs_callback(self, left_img_msg, right_img_msg, chest_img_msg, depth_img_msg):
        # Motor Trigger Sync (3.33 FPS or 299.99 ms)
        max_stamp = max(left_img_msg.header.stamp, right_img_msg.header.stamp, chest_img_msg.header.stamp)
        elapsed_time = (max_stamp - self.frame_stamp_tminus1).to_sec()
        if elapsed_time > 500e-3:
            print('--------------')
            rospy.loginfo(f'FPS: {1/elapsed_time: .{2}f}')
            self.frame_stamp_tminus1 = max_stamp

            # Initialization
            dx_l, dy_l, dx_r, dy_r = 0, 0, 0, 0
            theta_l_pan, theta_r_pan = None, None
            theta_l_tilt, theta_r_tilt, theta_tilt = None, None, None

            # Conversion of ROS Message
            self.left_img = self.bridge.imgmsg_to_cv2(left_img_msg, "bgr8")
            self.right_img = self.bridge.imgmsg_to_cv2(right_img_msg, "bgr8")
            self.chest_img = self.bridge.imgmsg_to_cv2(chest_img_msg, "bgr8")
            self.depth_img = self.bridge.imgmsg_to_cv2(depth_img_msg, "16UC1")
            depth_map_normalized = copy.deepcopy(self.depth_img) / self.depth_img.max()  # Normalize values to [0, 1]
            gray_depth_img = (depth_map_normalized * 255).astype(np.uint8)
            gray_depth_img = cv2.cvtColor(gray_depth_img, cv2.COLOR_GRAY2BGR)
            # print(left_img_msg.header, right_img_msg.header, chest_img_msg.header)
            # print(self.depth_img)

            ## Attention ##
            
            # Pan Target
            if self.ctr%2 == 0:
                self.chess_idx = random.randint(0,53)
            print('Chess Idx:', self.chess_idx)
            
            # Process Left Eye, Right Eye, Chest Cam Target
            left_eye_pxs = self.attention.process_img(self.chess_idx, copy.deepcopy(self.left_img))
            right_eye_pxs = self.attention.process_img(self.chess_idx, copy.deepcopy(self.right_img))
            chest_cam_pxs = self.attention.process_img(self.chess_idx, copy.deepcopy(self.chest_img))

            # Calculate Delta between Gaze Center and Pixel Target
            if chest_cam_pxs is None:
                left_eye_px = (-self.calib_params['left_eye']['x_center'], -self.calib_params['left_eye']['y_center'])
                right_eye_px = (-self.calib_params['right_eye']['x_center'], -self.calib_params['right_eye']['y_center'])
                chest_cam_px = (424, 240)
                left_eye_px_tminus1 = left_eye_px
                right_eye_px_tminus1 = right_eye_px
                chest_cam_px_tminus1 = chest_cam_px
            elif left_eye_pxs is None or right_eye_pxs is None:
                left_eye_px = (-self.calib_params['left_eye']['x_center'], -self.calib_params['left_eye']['y_center'])
                right_eye_px = (-self.calib_params['right_eye']['x_center'], -self.calib_params['right_eye']['y_center'])
                chest_cam_px = tuple(chest_cam_pxs[self.chess_idx].tolist())
                left_eye_px_tminus1 = copy.deepcopy(left_eye_px)
                right_eye_px_tminus1 = copy.deepcopy(right_eye_px)
                chest_cam_px_tminus1 = chest_cam_pxs[self.chess_idx_tminus1]
            else:
                # Preprocessing
                left_eye_px = tuple(left_eye_pxs[self.chess_idx].tolist())
                right_eye_px = tuple(right_eye_pxs[self.chess_idx].tolist())
                chest_cam_px = tuple(chest_cam_pxs[self.chess_idx].tolist())
                left_eye_px_tminus1 = left_eye_pxs[self.chess_idx_tminus1]
                right_eye_px_tminus1 = right_eye_pxs[self.chess_idx_tminus1]
                chest_cam_px_tminus1 = chest_cam_pxs[self.chess_idx_tminus1]


            ## Geometric Intersection
            x,y,z = self.depth_to_pointcloud(chest_cam_px, self.depth_img, self.camera_mtx['chest_cam']['camera_matrix'], z_replace=1.0)
            pts = self.transform_points(np.array([[x,y,z]]), np.array(self.calib_params['transformations']["T_origin_chest"])).squeeze()
            T_origin_left_eye_ctr = (np.array(self.calib_params['transformations']["T_origin_chest"]) 
                                 @ np.array(self.calib_params['transformations']["T_chest_left_eye"]) 
                                 @ np.array(self.calib_params['transformations']["T_left_eye_gaze_ctr"]))
            T_origin_right_eye_ctr = (np.array(self.calib_params['transformations']["T_origin_chest"]) 
                                  @ np.array(self.calib_params['transformations']["T_chest_right_eye"])
                                  @ np.array(self.calib_params['transformations']["T_right_eye_gaze_ctr"]))
            
            # OpenCV Orientation: x (to the right), y (to down), z (straight away from robot, depth)
            target_x = pts[0]
            target_y = pts[1]
            target_z = max(0.3, pts[2])
            target_pts = np.array([[target_x, target_y, target_z]])

            # Angles Calculation
            left_eye_pts = self.transform_points(target_pts, np.linalg.inv(T_origin_left_eye_ctr)).squeeze()
            right_eye_pts = self.transform_points(target_pts, np.linalg.inv(T_origin_right_eye_ctr)).squeeze()
            left_tilt = math.atan2(left_eye_pts[1], left_eye_pts[2])
            right_tilt = math.atan2(right_eye_pts[1], right_eye_pts[2])
            eyes_tilt = math.degrees((left_tilt + right_tilt)/2)  # OpenCV coordinates: CW is positive
            left_pan = math.degrees(math.atan2(left_eye_pts[0], left_eye_pts[2]))
            right_pan = math.degrees(math.atan2(right_eye_pts[0], right_eye_pts[2]))
            # rospy.loginfo(f"left_eye_pan (rad): {left_pan: .{4}f}")
            # rospy.loginfo(f"right_eye_pan (rad): {right_pan: .{4}f}")
            # rospy.loginfo(f"eyes_tilt (rad): {eyes_tilt: .{4}f}")
            
            # Publish Joint States
            if target_z != 0.3:
                joints = ['eyes_pitch']
                positions = [eyes_tilt]
                # self.publish_joint_state(joints, positions)
                joints = ['lefteye_yaw', 'righteye_yaw']
                positions = [left_pan, right_pan]
                # self.publish_joint_state(joints, positions)
                
                # Output of the Geometric Intersection
                theta_l_pan = left_pan/self.calib_params['left_eye']['slope']
                theta_r_pan = right_pan/self.calib_params['right_eye']['slope']
                theta_tilt = eyes_tilt/self.calib_params['tilt_eyes']['slope']

            # Visualize the Previous Target          
            left_img = cv2.drawMarker(copy.deepcopy(self.left_img), (round(left_eye_px_tminus1[0]),round(left_eye_px_tminus1[1])), color=(0, 0, 255), 
                                markerType=cv2.MARKER_TILTED_CROSS, markerSize=13, thickness=2)
            right_img = cv2.drawMarker(copy.deepcopy(self.right_img), (round(right_eye_px_tminus1[0]),round(right_eye_px_tminus1[1])), color=(0, 0, 255), 
                        markerType=cv2.MARKER_TILTED_CROSS, markerSize=13, thickness=2)
            chest_img = cv2.drawMarker(copy.deepcopy(self.chest_img), (round(chest_cam_px_tminus1[0]),round(chest_cam_px_tminus1[1])), color=(0, 0, 255), 
                        markerType=cv2.MARKER_TILTED_CROSS, markerSize=13, thickness=2)
            
            # Dx and Dy for Left and Right
            dx_l = left_eye_px_tminus1[0] - self.calib_params['left_eye']['x_center']
            dy_l = self.calib_params['left_eye']['y_center'] - left_eye_px_tminus1[1]
            dx_r = right_eye_px_tminus1[0] - self.calib_params['right_eye']['x_center']
            dy_r = self.calib_params['right_eye']['y_center'] - right_eye_px_tminus1[1]

            # Storing
            with self.buffer_lock:
                # Get Motor State
                with self.motor_lock:
                    self.calibration.store_latest_state(self._motor_states)

                    self.rl_state['chess_idx'] = copy.deepcopy(self.chess_idx)
                    self.rl_state['theta_left_pan'] = self._motor_states[0]['angle']
                    self.rl_state['theta_right_pan'] = self._motor_states[1]['angle']
                    self.rl_state['theta_tilt'] = self._motor_states[2]['angle']
                    self.rl_state['chest_cam_px_x_tminus1'] = chest_cam_px_tminus1[0]
                    self.rl_state['chest_cam_px_y_tminus1'] = chest_cam_px_tminus1[1]
                    self.rl_state['left_eye_px_x_tminus1'] = left_eye_px_tminus1[0]
                    self.rl_state['left_eye_px_y_tminus1'] = left_eye_px_tminus1[1]
                    self.rl_state['right_eye_px_x_tminus1'] = right_eye_px_tminus1[0]
                    self.rl_state['right_eye_px_y_tminus1'] = right_eye_px_tminus1[1]
                    self.rl_state['chest_cam_px_x'] = chest_cam_px[0]
                    self.rl_state['chest_cam_px_y'] = chest_cam_px[1]
                    self.rl_state['left_eye_px_x'] = left_eye_px[0]
                    self.rl_state['left_eye_px_y'] = left_eye_px[1]
                    self.rl_state['right_eye_px_x'] = right_eye_px[0]
                    self.rl_state['right_eye_px_y'] = right_eye_px[1]
                    self.rl_state['dx_l'] = dx_l
                    self.rl_state['dy_l'] = dy_l
                    self.rl_state['dx_r'] = dx_r
                    self.rl_state['dy_r'] = dy_r    

                    self.rl_state['3d_point'] = (target_x, target_y, target_z)
                    self.rl_state['chest_pan_angle'] =  math.atan2(target_x, target_z)
                    self.rl_state['chest_tilt_angle'] =  math.atan2(target_y, target_z)
                    
                    self.rl_state['plan_phi_left_pan'] = left_pan
                    self.rl_state['plan_phi_right_pan'] = right_pan
                    self.rl_state['plan_phi_tilt'] = eyes_tilt

                    self.rl_state['chest_img'] = self.chest_img
                    self.rl_state['left_eye_img'] = self.left_img
                    self.rl_state['right_eye_img'] = self.right_img
                    self.rl_state['depth_img'] = self.depth_img

                    self.rl_state['chest_img_stamp'] = chest_img_msg.header.stamp.to_sec()
                    self.rl_state['left_eye_img_stamp'] = left_img_msg.header.stamp.to_sec()
                    self.rl_state['right_eye_img_stamp'] = right_img_msg.header.stamp.to_sec()
                    self.rl_state['depth_img_stamp'] = depth_img_msg.header.stamp.to_sec()
                    # print(self.rl_state)

                    # CSV Save
                    self.state_list['chess_idx'].append(copy.deepcopy(self.chess_idx))
                    self.state_list['theta_left_pan'].append(self._motor_states[0]['angle'])
                    self.state_list['theta_right_pan'].append(self._motor_states[1]['angle'])
                    self.state_list['theta_tilt'].append(self._motor_states[2]['angle'])
                    self.state_list['chest_cam_px_x_tminus1'].append(chest_cam_px_tminus1[0])
                    self.state_list['chest_cam_px_y_tminus1'].append(chest_cam_px_tminus1[1])
                    self.state_list['left_eye_px_x_tminus1'].append(left_eye_px_tminus1[0])
                    self.state_list['left_eye_px_y_tminus1'].append(left_eye_px_tminus1[1])
                    self.state_list['right_eye_px_x_tminus1'].append(right_eye_px_tminus1[0])
                    self.state_list['right_eye_px_y_tminus1'].append(right_eye_px_tminus1[1])
                    self.state_list['chest_cam_px_x'].append(chest_cam_px[0])
                    self.state_list['chest_cam_px_y'].append(chest_cam_px[1])
                    self.state_list['left_eye_px_x'].append(left_eye_px[0])
                    self.state_list['left_eye_px_y'].append(left_eye_px[1])
                    self.state_list['right_eye_px_x'].append(right_eye_px[0])
                    self.state_list['right_eye_px_y'].append(right_eye_px[1])

                    self.state_list['dx_l'].append(dx_l)
                    self.state_list['dy_l'].append(dy_l)
                    self.state_list['dx_r'].append(dx_r)
                    self.state_list['dy_r'].append(dy_r)

                    self.state_list['3d_point'].append((target_x, target_y, target_z))
                    self.state_list['chest_pan_angle'].append(math.atan2(target_x, target_z))
                    self.state_list['chest_tilt_angle'].append(math.atan2(target_y, target_z))

                    self.state_list['plan_phi_left_pan'].append(left_pan)
                    self.state_list['plan_phi_right_pan'].append(right_pan)
                    self.state_list['plan_phi_tilt'].append(eyes_tilt)

                    self.state_list['chest_img_stamp'].append(chest_img_msg.header.stamp.to_sec())
                    self.state_list['left_eye_img_stamp'].append(left_img_msg.header.stamp.to_sec())
                    self.state_list['right_eye_img_stamp'].append(right_img_msg.header.stamp.to_sec())
                    self.state_list['depth_img_stamp'].append(depth_img_msg.header.stamp.to_sec())

            # Movement
            # theta_l_pan, theta_r_pan, theta_tilt = None, None, None
            # Wait for the new command

            with self.action_lock:
                if self.action != None:
                    theta_l_pan = self.action[0]
                    theta_r_pan = self.action[1]
                    theta_tilt = self.action[2]

            if (theta_l_pan is not None) or (theta_r_pan is not None) or (theta_tilt is not None):
                
                if self.ctr % 2 == 1:
                    theta_l_pan = theta_l_pan
                    theta_r_pan = theta_r_pan
                    theta_tilt = -theta_tilt
                else:
                    theta_l_pan = -18
                    theta_r_pan = -18
                    theta_tilt = 22
                
                # print('debug:', theta_l_pan)
                print('theta_l_pan:', theta_l_pan)
                print('theta_r_pan:', theta_r_pan)
                print('theta_tilt:', theta_tilt)
                self.move([theta_l_pan, theta_r_pan, theta_tilt])
                with self.buffer_lock:
                    self.rl_state['theta_left_pan_cmd'] = theta_l_pan
                    self.rl_state['theta_right_pan_cmd'] = theta_r_pan 
                    self.rl_state['theta_tilt_cmd'] = theta_tilt

                    self.state_list['theta_left_pan_cmd'].append(theta_l_pan)
                    self.state_list['theta_right_pan_cmd'].append(theta_r_pan)
                    self.state_list['theta_tilt_cmd'].append(theta_tilt)

                    self.pickle_data['data'].append(copy.deepcopy(self.rl_state))

            # Visualization
            left_img = self.ctr_cross_img(left_img, 'left_eye')
            right_img = self.ctr_cross_img(right_img, 'right_eye')
            chest_img = self.ctr_cross_img(chest_img, 'chest_cam')
            concat_img = np.hstack((chest_img, left_img, right_img))
            
            # Resizing
            height, width = concat_img.shape[:2]
            concat_img = cv2.resize(concat_img, (round(width/2), round(height/2)))

            # Output Display 1
            self.rt_display_pub.publish(self.bridge.cv2_to_imgmsg(concat_img, encoding="bgr8"))

            # Reassigning
            self.chess_idx_tminus1 = copy.deepcopy(self.chess_idx)
    
            # Saving
            self.ctr+=1
            rospy.loginfo("===%i/%i" % (self.ctr, self.num_ctr))
            if self.ctr==self.num_ctr:
                
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                results_dir = os.path.join(parent_dir, 'results','pantilt_baseline_eval')
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                    print(f"Directory created: {results_dir}")
                else:
                    print(f"Directory already exists: {results_dir}")
                
                dt_str = datetime.now().strftime("_%Y%m%d_%H%M%S_%f")
                title_str = 'pantilt_baseline_eval'
                pickle_path = os.path.join(results_dir, title_str+'_pickle'+dt_str+'.pickle')

                # Saving to Pickle File
                with open(pickle_path, 'wb') as file:
                    pickle.dump(self.pickle_data, file)
                print('=====================')
                print('Pickle file saved in:', pickle_path)

                # Saving CSV
                df = pd.DataFrame(self.state_list)

                # Save DataFrame as CSV
                csv_path = os.path.join(results_dir, title_str+'_csv'+dt_str+'.csv')
                df.to_csv(csv_path, index=False)
                print('CSV file saved:',csv_path)

                end = time.time()
                print('[ELAPSED_TIME]', (end-self.start), 'secs')
                rospy.signal_shutdown('End')
                sys.exit()

    def _capture_limits(self, motor):
        int_min = motors_dict[motor]['motor_min']
        int_init = motors_dict[motor]['init']
        int_max = motors_dict[motor]['motor_max']
        angle_min = motor_int_to_angle(motor, int_min, self.degrees)
        angle_init = motor_int_to_angle(motor, int_init, self.degrees)
        angle_max = motor_int_to_angle(motor, int_max, self.degrees)
        limits = {'int_min': int_min, 
                  'int_init': int_init, 
                  'int_max': int_max,
                  'angle_min': angle_min, 
                  'angle_init': angle_init, 
                  'angle_max': angle_max}
        return limits

    def _check_limits(self, name, value):
        if value < self._motor_limits[name]['angle_min']:
            value = self._motor_limits[name]['angle_min']
        elif value > self._motor_limits[name]['angle_max']:
            value = self._motor_limits[name]['angle_max']
        return value

    def _set_motor_limits(self, names: list):
        self.names = names
        self.num_names = len(self.names)
        self._motor_limits = {motor: self._capture_limits(motor) for motor in self.names}
        self._motor_states = [None]*self.num_names
    
    def move(self, values):
        values = [self._check_limits(self.names[i],x) for i,x in enumerate(values)]
        if self.degrees:
            values = [math.radians(x) for x in values]
        args = {"names":self.names, "values":values}
        self.motor_pub.publish(TargetPosture(**args))

    def move_specific(self, names, values):
        values = [self._check_limits(names[i],x) for i,x in enumerate(values)]
        if self.degrees:
            values = [math.radians(x) for x in values]
        args = {"names":names, "values":values}
        self.motor_pub.publish(TargetPosture(**args))

    def ctr_cross_img(self, img, eye:str):
        """eye (str): select from ['left_eye', 'right_eye', 'chest_cam']
        """
        if eye == 'chest_cam':
            # True Optical Center
            img = cv2.putText(img, eye, (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                              1, (0,255,0), 2, cv2.LINE_AA)
            img = cv2.line(img, (round(self.camera_mtx_params[eye]['cx']), 0), (round(self.camera_mtx_params[eye]['cx']), 480), (0,255,0))
            img = cv2.line(img, (0, round(self.camera_mtx_params[eye]['cy'])), (848, round(self.camera_mtx_params[eye]['cy'])), (0,255,0))
            img = cv2.drawMarker(img, (round(self.camera_mtx_params[eye]['cx']), round(self.camera_mtx_params[eye]['cy'])), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        else:
            img = cv2.putText(img, eye, (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                              1, (0,255,0), 2, cv2.LINE_AA)
            img = cv2.line(img, (round(self.calib_params[eye]['x_center']), 0), (round(self.calib_params[eye]['x_center']), 480), (0,255,0))
            img = cv2.line(img, (0, round(self.calib_params[eye]['y_center'])), (640, round(self.calib_params[eye]['y_center'])), (0,255,0))
            img = cv2.drawMarker(img, (round(self.calib_params[eye]['x_center']), round(self.calib_params[eye]['y_center'])), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        return img


if __name__ == '__main__':
    rospy.init_node('visuomotor')
    vismotor = VisuoMotorNode(num_trials=600)
    rospy.spin()
