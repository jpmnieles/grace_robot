import os
import sys
sys.path.append(os.getcwd())

import math
import time
import yaml
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

    joints_list = ['neck_roll', 'neck_pitch', 'neck_yaw',
                   'head_roll', 'head_pitch', 
                   'eyes_pitch', 'lefteye_yaw', 'righteye_yaw']


    def __init__(self, motors=["EyeTurnLeft", "EyeTurnRight", "EyesUpDown"], degrees=True):
        self.motor_lock = threading.Lock()
        self.buffer_lock = threading.Lock()
        self.action_lock = threading.Lock()
        self.motors = motors
        self.degrees = degrees
        self._set_motor_limits(motors)

        self.attention = ArucoAttention()
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
        if z==0:
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
        if elapsed_time > 285e-3:
            start = time.time()
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

            # Process Left Eye, Right Eye, Chest Cam Target
            left_eye_px, left_img = self.attention.process_img(copy.deepcopy(self.left_img),
                                                               np.array(self.camera_mtx['left_eye']['camera_matrix']),
                                                               np.array(self.camera_mtx['left_eye']['distortion_coefficients']))
            right_eye_px, right_img = self.attention.process_img(copy.deepcopy(self.right_img),
                                                               np.array(self.camera_mtx['right_eye']['camera_matrix']),
                                                               np.array(self.camera_mtx['right_eye']['distortion_coefficients']))
            chest_cam_px, chest_img = self.attention.process_img(copy.deepcopy(self.chest_img),
                                                               np.array(self.camera_mtx['chest_cam']['camera_matrix']),
                                                               np.array(self.camera_mtx['chest_cam']['distortion_coefficients']))

            print('Target_pts:', chest_cam_px)
            
            # Calculate Delta between Gaze Center and Pixel Target
            if chest_cam_px is None:
                left_eye_px = (-self.calib_params['left_eye']['x_center'], -self.calib_params['left_eye']['y_center'])
                right_eye_px = (-self.calib_params['right_eye']['x_center'], -self.calib_params['right_eye']['y_center'])
                chest_cam_px = (240, 200)
            elif left_eye_px is None or right_eye_px is None:
                left_eye_px = (-self.calib_params['left_eye']['x_center'], -self.calib_params['left_eye']['y_center'])
                right_eye_px = (-self.calib_params['right_eye']['x_center'], -self.calib_params['right_eye']['y_center'])
            elif chest_cam_px[0] >= 848 or chest_cam_px[1] >= 480:
                chest_cam_px = (847, 479)
            
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
            print('Target Pts:', target_pts)

            # Angles Calculation
            left_eye_pts = self.transform_points(target_pts, np.linalg.inv(T_origin_left_eye_ctr)).squeeze()
            right_eye_pts = self.transform_points(target_pts, np.linalg.inv(T_origin_right_eye_ctr)).squeeze()
            left_tilt = math.atan2(left_eye_pts[1], left_eye_pts[2])
            right_tilt = math.atan2(right_eye_pts[1], right_eye_pts[2])
            eyes_tilt = math.degrees((left_tilt + right_tilt)/2.0)  # OpenCV coordinates: CW is positive
            left_pan = math.degrees(math.atan2(left_eye_pts[0], left_eye_pts[2]))
            right_pan = math.degrees(math.atan2(right_eye_pts[0], right_eye_pts[2]))
            
            # Publish Joint States
            if target_z != 0.3:                
                # Output of the Geometric Intersection
                theta_l_pan = left_pan/self.calib_params['left_eye']['slope']
                theta_r_pan = right_pan/self.calib_params['right_eye']['slope']
                theta_tilt = -eyes_tilt/self.calib_params['tilt_eyes']['slope']

            with self.action_lock:
                if self.action != None:
                    theta_l_pan = self.action[0]
                    theta_r_pan = self.action[1]
                    theta_tilt = self.action[2]

            if (theta_l_pan is not None) or (theta_r_pan is not None) or (theta_tilt is not None):
                print('theta_l_pan:', theta_l_pan)
                print('theta_r_pan:', theta_r_pan)
                print('theta_tilt:', theta_tilt)
                self.move([theta_l_pan, theta_r_pan, theta_tilt])

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
    vismotor = VisuoMotorNode()
    rospy.spin()
