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
        'theta_left_pan': None,
        'theta_right_pan': None,
        'theta_tilt': None,
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

    motor_state = {
        "EyeTurnLeft": None,
        "EyeTurnRight": None, 
        "EyesUpDown": None,
        "NeckRotation": None,
        "UpperGimbalLeft": None,
        "UpperGimbalRight": None,
        "LowerGimbalLeft": None,
        "LowerGimbalRight": None
    }
    
    pickle_data = {
        'subject_num': None,
        'markers': None,
        'trial_num': None,
        'data': [],
    }

    def __init__(self, motors=["EyeTurnLeft", "EyeTurnRight", "EyesUpDown",
                               "NeckRotation", "UpperGimbalLeft", "UpperGimbalRight",
                               "LowerGimbalLeft", "LowerGimbalRight"], degrees=True):
        # Timer Start
        self.start = time.time()
        rospy.loginfo('Starting')

        # Locks
        self.motor_lock = threading.Lock()
        self.buffer_lock = threading.Lock()
        self.left_cam_lock = threading.Lock()
        self.right_cam_lock = threading.Lock()
        self.chest_cam_lock = threading.Lock()
        self.depth_cam_lock = threading.Lock()

        # Initialization
        self.motors = motors
        self.degrees = degrees
        self._set_motor_limits(motors)
        self.camera_mtx = load_camera_mtx()
        self.calib_params = load_json('config/calib/calib_params.json')
        self.motor_pub = rospy.Publisher('/hr/actuators/pose', TargetPosture, queue_size=1)
        self.rt_display_pub = rospy.Publisher('/output_display1', Image, queue_size=1)
        self.bridge = CvBridge()
        time.sleep(1)
        
        # Initial Reset Action
        self.move((0, 0, 0, 0, 44, -44, -13, 13))
        time.sleep(3.0)
        self.move((0, 0, 0, 0, 0, 0, 0, 0))
        time.sleep(3.0)
        self.move((0, 0, 0, -40, 0, 0, 0, 0))
        time.sleep(3.0)
        self.move((0, 0, 0, 0, 0, 0, 0, 0))
        time.sleep(3.0)
        self.move((-18, -18, 22))
        time.sleep(1.0)
        self.move((0, 0, 0))
        time.sleep(1.0)

        # Subscribers
        self.motor_sub = rospy.Subscriber('/hr/actuators/motor_states', MotorStateList, self._capture_state)
        self.left_cam_sub = rospy.Subscriber('/left_eye/image_raw', Image, self.left_img_callback)
        self.right_cam_sub = rospy.Subscriber('/right_eye/image_raw', Image, self.right_img_callback)
        self.chest_cam_sub = rospy.Subscriber('/hr/perception/jetson/realsense/camera/color/image_raw', 
                                              Image, self.chest_img_callback)
        self.depth_cam_sub = rospy.Subscriber('/hr/perception/jetson/realsense/camera/aligned_depth_to_color/image_raw', 
                                              Image, self.depth_img_callback)
        time.sleep(3.0)

    def _capture_state(self, msg):
        """Callback for capturing motor state
        """
        self._msg = msg
        with self.motor_lock:
            for idx, motor_msg in enumerate(msg.motor_states):
                if motor_msg.name in self.names:
                    idx = self.motors.index(motor_msg.name)
                    self._motor_states[idx] = message_converter.convert_ros_message_to_dictionary(motor_msg)
                    self._motor_states[idx]['angle'] = motor_int_to_angle(motor_msg.name, motor_msg.position, self.degrees)
      
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

    def left_img_callback(self, left_img_msg):
        with self.left_cam_lock:
            self.left_img = self.bridge.imgmsg_to_cv2(left_img_msg, "bgr8")
    
    def right_img_callback(self, right_img_msg):
        with self.right_cam_lock:
            self.right_img = self.bridge.imgmsg_to_cv2(right_img_msg, "bgr8")

    def chest_img_callback(self, chest_img_msg):
        with self.chest_cam_lock:
            self.chest_img = self.bridge.imgmsg_to_cv2(chest_img_msg, "bgr8")

    def depth_img_callback(self, depth_img_msg):
        with self.depth_cam_lock:
            self.depth_raw = self.bridge.imgmsg_to_cv2(depth_img_msg, "16UC1")
            depth_norm = 255*(copy.deepcopy(self.depth_raw) / self.depth_raw.max())
            depth_norm = depth_norm.astype(np.uint8)
            self.depth_img = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    def eye_imgs_callback(self, left_img_msg, right_img_msg, chest_img_msg, depth_img_msg):
        # Motor Trigger Sync (3.33 FPS or 299.99 ms)
        max_stamp = max(left_img_msg.header.stamp, right_img_msg.header.stamp, chest_img_msg.header.stamp)
        elapsed_time = (max_stamp - self.frame_stamp_tminus1).to_sec()
        if elapsed_time > 1000e-3:
            print('--------------')
            rospy.loginfo(f'FPS: {1/elapsed_time: .{2}f}')
            self.frame_stamp_tminus1 = max_stamp

            # Initialization
            theta_l_pan, theta_r_pan = None, None
            
            # Conversion of ROS Message
            self.left_img = self.bridge.imgmsg_to_cv2(left_img_msg, "bgr8")
            self.right_img = self.bridge.imgmsg_to_cv2(right_img_msg, "bgr8")
            self.chest_img = self.bridge.imgmsg_to_cv2(chest_img_msg, "bgr8")
            self.depth_img = self.bridge.imgmsg_to_cv2(depth_img_msg, "16UC1")
            depth_map_normalized = copy.deepcopy(self.depth_img) / self.depth_img.max()  # Normalize values to [0, 1]
            print(np.mean(self.depth_img)/1000.0)
            gray_depth_img = (depth_map_normalized * 255).astype(np.uint8)
            gray_depth_img = cv2.cvtColor(gray_depth_img, cv2.COLOR_GRAY2BGR)

            ## Attention ##
            
            # Pan Target
            self.chess_idx = self.chess_seq[self.ctr%self.num_chess_seq]
            # print(self.chess_idx)
            
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

            # Visualize the Previous Target
            left_img = cv2.drawMarker(copy.deepcopy(self.left_img), (round(left_eye_px[0]),round(left_eye_px[1])), color=(204, 41, 204), 
                                markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)
            left_img = cv2.drawMarker(left_img, (round(left_eye_px_tminus1[0]),round(left_eye_px_tminus1[1])), color=(0, 0, 255), 
                                markerType=cv2.MARKER_TILTED_CROSS, markerSize=13, thickness=2)
            right_img = cv2.drawMarker(copy.deepcopy(self.right_img), (round(right_eye_px[0]),round(right_eye_px[1])), color=(204, 41, 204), 
                        markerType=cv2.MARKER_STAR, markerSize=13, thickness=2)
            right_img = cv2.drawMarker(right_img, (round(right_eye_px_tminus1[0]),round(right_eye_px_tminus1[1])), color=(0, 0, 255), 
                        markerType=cv2.MARKER_TILTED_CROSS, markerSize=13, thickness=2)
            chest_img = cv2.drawMarker(copy.deepcopy(self.chest_img), (round(chest_cam_px[0]),round(chest_cam_px[1])), color=(204, 41, 204), 
                        markerType=cv2.MARKER_STAR, markerSize=13, thickness=2)
            chest_img = cv2.drawMarker(chest_img, (round(chest_cam_px_tminus1[0]),round(chest_cam_px_tminus1[1])), color=(0, 0, 255), 
                        markerType=cv2.MARKER_TILTED_CROSS, markerSize=13, thickness=2)

            # Storing
            with self.buffer_lock:
                # Get Motor State
                with self.motor_lock:

                    self.rl_state['theta_left_pan'] = self._motor_states[0]['angle']
                    self.rl_state['theta_right_pan'] = self._motor_states[1]['angle']
                    self.rl_state['theta_tilt'] = self._motor_states[2]['angle']

                    self.rl_state['chest_img'] = self.chest_img
                    self.rl_state['left_eye_img'] = self.left_img
                    self.rl_state['right_eye_img'] = self.right_img
                    self.rl_state['depth_img'] = self.depth_img

                    self.rl_state['chest_img_stamp'] = chest_img_msg.header.stamp.to_sec()
                    self.rl_state['left_eye_img_stamp'] = left_img_msg.header.stamp.to_sec()
                    self.rl_state['right_eye_img_stamp'] = right_img_msg.header.stamp.to_sec()
                    self.rl_state['depth_img_stamp'] = depth_img_msg.header.stamp.to_sec()
                    # print(self.rl_state)

            # Wait for the new command
            theta_l_pan_list = list(range(-18,19,2))
            theta_r_pan_list = list(range(-18,19,2))
            theta_tilt_list = list(range(20,-31,-5))
            repetition = 5
            
            theta_tilt_ovr = theta_tilt_list[((self.ctr//2)//(len(theta_l_pan_list)*repetition))%(len(theta_tilt_list))]
            
            if self.ctr % 2 == 1:
                theta_l_pan = theta_l_pan_list[(self.ctr//2)%(len(theta_l_pan_list))]
                theta_r_pan = theta_r_pan_list[(self.ctr//2)%(len(theta_r_pan_list))]
            else:
                theta_l_pan = -18
                theta_r_pan = -18
                theta_tilt_ovr = 22
            
            print('debug:', theta_l_pan, theta_r_pan, theta_tilt_ovr)
            
            self.move((theta_l_pan, theta_r_pan, theta_tilt_ovr))
            with self.buffer_lock:
                self.rl_state['theta_left_pan_cmd'] = theta_l_pan
                self.rl_state['theta_right_pan_cmd'] = theta_r_pan 
                self.rl_state['theta_tilt_cmd'] = theta_tilt_ovr
                self.pickle_data['data'].append(copy.deepcopy(self.rl_state))

            # Visualization
            left_img = self.ctr_cross_img(left_img, 'left_eye')
            right_img = self.ctr_cross_img(right_img, 'right_eye')
            chest_img = self.ctr_cross_img(chest_img, 'chest_cam')
            concat_img = np.hstack((chest_img, left_img, right_img, gray_depth_img))
            
            # Resizing
            height, width = concat_img.shape[:2]
            concat_img = cv2.resize(concat_img, (round(width/2), round(height/2)))

            # Output Display 1
            self.rt_display_pub.publish(self.bridge.cv2_to_imgmsg(concat_img, encoding="bgr8"))

            # Reassigning
            self.chess_idx_tminus1 = self.chess_idx
    
            # Saving
            self.ctr+=1
            rospy.loginfo("===%i/%i" % (self.ctr, self.num_ctr))
            if self.ctr==self.num_ctr:
                
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                results_dir = os.path.join(parent_dir, 'results','pantilt_baseline_sweep_charuco')
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                    print(f"Directory created: {results_dir}")
                else:
                    print(f"Directory already exists: {results_dir}")
                
                dt_str = datetime.now().strftime("_%Y%m%d_%H%M%S_%f")
                title_str = 'pantilt_baseline_sweep_charuco'
                pickle_path = os.path.join(results_dir, title_str+'_pickle'+dt_str+'.pickle')

                # Saving to Pickle File
                with open(pickle_path, 'wb') as file:
                    pickle.dump(self.pickle_data, file)
                print('=====================')
                print('Pickle file saved in:', pickle_path)

                end = time.time()
                print('[ELAPSED_TIME]', (end-self.start), 'secs')
                rospy.signal_shutdown('End')
                sys.exit()

    def run(self):
        rospy.loginfo('Running')

        # List of Motor Commands
        list_ep = list(range(-14,15,2))
        list_et = list(range(20,-31,-5))
        list_lnp = list(range(-35,36,5))
        list_lnt = list(range(-10,31,10))
        list_unt = list(range(40,-11,-10))

        # Get Images
        global left_img, left_img, right_img, chest_img
        global depth_raw, depth_img

        # Neck
        for lnt in list_lnt:
            for unt in list_unt:
                for i in range(2):
                    if i==0:
                        self.move_specific(["UpperGimbalLeft","UpperGimbalRight","LowerGimbalLeft","LowerGimbalRight"],
                                           [44,-44,-13,13])
                        rospy.loginfo('lnt & unt reset')
                        rospy.sleep(3)
                    else:
                        self.move_specific(["UpperGimbalLeft","UpperGimbalRight","LowerGimbalLeft","LowerGimbalRight"],
                                           [unt,-unt,-lnt,lnt])
                        rospy.loginfo('lnt:%d, unt:%d' % (lnt,unt))
                        rospy.sleep(3)
                    
                # Neck Rotation
                for lnp in list_lnp:
                    for j in range(2):
                        if j==0:
                            self.move_specific(["NeckRotation"],[-40])
                            rospy.loginfo('lnp reset')
                            rospy.sleep(3)
                        else:
                            self.move_specific(["NeckRotation"],[lnp])
                            rospy.loginfo('lnp:%d' % (lnp))
                            rospy.sleep(3)

                    # Eyes
                    for et in list_et:
                        for ep in list_ep:             
                            for k in range(2):
                                if k==0:
                                    # Reset
                                    self.move_specific(["EyeTurnLeft","EyeTurnRight","EyesUpDown"],
                                                        [-18,-18,22])
                                    rospy.loginfo('et & ep reset')
                                    rospy.sleep(1)
                                else:
                                    self.move_specific(["EyeTurnLeft", "EyeTurnRight", "EyesUpDown"],
                                                        [ep,ep,et])
                                    rospy.loginfo('et:%d, ep:%d' % (et,ep))
                                    rospy.sleep(1)
                                
                            # Capture
                            with self.left_cam_lock:
                                left_img = copy.deepcopy(self.left_img)
                            with self.right_cam_lock:
                                right_img = copy.deepcopy(self.right_img)
                            with self.chest_cam_lock:
                                chest_img = copy.deepcopy(self.chest_img)
                            with self.depth_cam_lock:
                                depth_raw = copy.deepcopy(self.depth_raw)
                                depth_img = copy.deepcopy(self.depth_img)
                            with self.motor_lock:
                                motor_state = [self._motor_states[i]['angle'] 
                                                for i in range(len(self._motor_states))]
                                print(self.motors)
                                print(motor_state)

                            # Visualization
                            concat_img = np.hstack((chest_img, left_img, right_img, depth_img))
                            height, width = concat_img.shape[:2]
                            concat_img = cv2.resize(concat_img, (round(width/2), round(height/2)))

                            # Output Display 1
                            self.rt_display_pub.publish(self.bridge.cv2_to_imgmsg(concat_img, encoding="bgr8"))
        
    
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
    
    def move_specific(self, names:list, values:list):
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
            img = cv2.line(img, (round(self.camera_mtx[eye]['cx']), 0), (round(self.camera_mtx[eye]['cx']), 480), (0,255,0))
            img = cv2.line(img, (0, round(self.camera_mtx[eye]['cy'])), (848, round(self.camera_mtx[eye]['cy'])), (0,255,0))
            img = cv2.drawMarker(img, (round(self.camera_mtx[eye]['cx']), round(self.camera_mtx[eye]['cy'])), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        else:
            img = cv2.line(img, (round(self.calib_params[eye]['x_center']), 0), (round(self.calib_params[eye]['x_center']), 480), (0,255,0))
            img = cv2.line(img, (0, round(self.calib_params[eye]['y_center'])), (640, round(self.calib_params[eye]['y_center'])), (0,255,0))
            img = cv2.drawMarker(img, (round(self.calib_params[eye]['x_center']), round(self.calib_params[eye]['y_center'])), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        return img


if __name__ == '__main__':
    rospy.init_node('visuomotor')
    vismotor = VisuoMotorNode()  # Manual Calculation (19(pan)x11(tilt)x2x2(reps)+2)
    vismotor.run()
