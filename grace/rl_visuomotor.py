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
from std_msgs.msg import Float32, String, Header
from cv_bridge import CvBridge, CvBridgeError

import cv2
import dlib
import numpy as np
import random
import datetime
import threading
import copy

from grace.utils import *
from aec.baseline import BaselineCalibration
from grace.attention import *


class VisuoMotorNode(object):
    
    state_buffer = {
        't-1': {
            'chest_cam_px': None,
            'left_eye_px': None,
            'right_eye_px': None,
            'theta_left_pan': None,
            'theta_right_pan': None,
            'theta_tilt': None,
        },
        't': {
            'chest_cam_px': None,
            'left_eye_px': None,
            'right_eye_px': None,
            'theta_left_pan': None,
            'theta_right_pan': None,
        }
    }

    camera_buffer = {
        't-1': {
            'left_eye': np.zeros((480,640,3), dtype=np.uint8),
            'right_eye': np.zeros((480,640,3), dtype=np.uint8),
            'chest_cam': np.zeros((480,848,3), dtype=np.uint8),
        },
        't': {
            'left_eye': np.zeros((480,640,3), dtype=np.uint8),
            'right_eye': np.zeros((480,640,3), dtype=np.uint8),
            'chest_cam': np.zeros((480,848,3), dtype=np.uint8),
        },
    }


    def __init__(self, motors=["EyeTurnLeft", "EyeTurnRight", "EyesUpDown"], degrees=True):
        self.motor_lock = threading.Lock()
        self.buffer_lock = threading.Lock()
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
        self.camera_mtx = load_camera_mtx()
        self.bridge = CvBridge()
        time.sleep(1)

        self.motor_sub = rospy.Subscriber('/hr/actuators/motor_states', MotorStateList, self._capture_state)
        self.left_eye_sub = message_filters.Subscriber("/left_eye/image_raw", Image)
        self.right_eye_sub = message_filters.Subscriber("/right_eye/image_raw", Image)  # TODO: change to right eye when there is better camera
        self.chest_cam_sub = message_filters.Subscriber('/hr/perception/jetson/realsense/camera/color/image_raw', Image)
        self.ats = message_filters.ApproximateTimeSynchronizer([self.left_eye_sub, self.right_eye_sub, self.chest_cam_sub], queue_size=1, slop=0.15)
        self.ats.registerCallback(self.eye_imgs_callback)
        self.rt_display_pub = rospy.Publisher('/output_display1', Image, queue_size=1)
        self.state_pub = rospy.Publisher('/grace/state', String, queue_size=1)

        self.disp_img = np.zeros((480,640,3), dtype=np.uint8)
        self.calib_params = load_json('config/calib/calib_params.json')
        rospy.loginfo('Running')


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


    def eye_imgs_callback(self, left_img_msg, right_img_msg, chest_img_msg):
        # Motor Trigger Sync (3.33 FPS or 299.99 ms)
        max_stamp = max(left_img_msg.header.stamp, right_img_msg.header.stamp, chest_img_msg.header.stamp)
        elapsed_time = (max_stamp - self.frame_stamp_tminus1).to_sec()
        if elapsed_time > 285e-3:
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
            # print(left_img_msg.header, right_img_msg.header, chest_img_msg.header)

            # Snapshot
            self.camera_buffer['t-1']['left_eye'] = copy.deepcopy(self.camera_buffer['t']['left_eye'])
            self.camera_buffer['t']['left_eye'] = copy.deepcopy(self.left_img)
            self.camera_buffer['t-1']['right_eye'] = copy.deepcopy(self.camera_buffer['t']['right_eye'])
            self.camera_buffer['t']['right_eye'] = copy.deepcopy(self.right_img)
            self.camera_buffer['t-1']['chest_cam'] = copy.deepcopy(self.camera_buffer['t']['chest_cam'])
            self.camera_buffer['t']['chest_cam'] = copy.deepcopy(self.right_img)

            ## Attention ##
            
            # Random Target
            chess_idx = random.randint(0,53)
            
            # Process Left Eye, Right Eye, Chest Cam Target
            left_eye_pxs = self.attention.process_img(chess_idx, self.left_img)
            right_eye_pxs = self.attention.process_img(chess_idx, self.right_img)
            chest_cam_pxs = self.attention.process_img(chess_idx, self.chest_img)

            # Calculate Delta between Gaze Center and Pixel Target
            if left_eye_pxs is None or right_eye_pxs is None or chest_cam_pxs is None:
                dx_l, dy_l, dx_r, dy_r = 0, 0, 0, 0
                left_eye_px = (self.calib_params['left_eye']['x_center'], self.calib_params['left_eye']['y_center'])
                right_eye_px = (self.calib_params['right_eye']['x_center'], self.calib_params['right_eye']['y_center'])
                chest_cam_px = (0, 0)
                left_eye_px_tminus1 = left_eye_px
                right_eye_px_tminus1 = right_eye_px
                chest_cam_px_tminus1 = chest_cam_px
            else:
                left_eye_px = tuple(left_eye_pxs[chess_idx].tolist())
                right_eye_px = tuple(right_eye_pxs[chess_idx].tolist())
                chest_cam_px = tuple(chest_cam_pxs[chess_idx].tolist())
                left_eye_px_tminus1 = left_eye_pxs[self.chess_idx_tminus1]
                right_eye_px_tminus1 = right_eye_pxs[self.chess_idx_tminus1]
                chest_cam_px_tminus1 = chest_cam_pxs[self.chess_idx_tminus1]

                dx_l = left_eye_px[0] - self.calib_params['left_eye']['x_center']
                dy_l = self.calib_params['left_eye']['y_center'] - left_eye_px[1]
                dx_r = right_eye_px[0] - self.calib_params['right_eye']['x_center']
                dy_r = self.calib_params['right_eye']['y_center'] - right_eye_px[1]

            # Visualize the Previous Target
            left_img = cv2.drawMarker(self.left_img, (round(left_eye_px[0]),round(left_eye_px[1])), color=(255, 0, 0), 
                                markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
            left_img = cv2.drawMarker(left_img, (round(left_eye_px_tminus1[0]),round(left_eye_px_tminus1[1])), color=(0, 0, 255), 
                                markerType=cv2.MARKER_TILTED_CROSS, markerSize=13, thickness=2)
            right_img = cv2.drawMarker(self.right_img, (round(right_eye_px[0]),round(right_eye_px[1])), color=(255, 0, 0), 
                        markerType=cv2.MARKER_CROSS, markerSize=13, thickness=2)
            right_img = cv2.drawMarker(right_img, (round(right_eye_px_tminus1[0]),round(right_eye_px_tminus1[1])), color=(0, 0, 255), 
                        markerType=cv2.MARKER_TILTED_CROSS, markerSize=13, thickness=2)
            chest_img = cv2.drawMarker(self.chest_img, (round(chest_cam_px[0]),round(chest_cam_px[1])), color=(255, 0, 0), 
                        markerType=cv2.MARKER_CROSS, markerSize=13, thickness=2)
            chest_img = cv2.drawMarker(chest_img, (round(chest_cam_px_tminus1[0]),round(chest_cam_px_tminus1[1])), color=(0, 0, 255), 
                        markerType=cv2.MARKER_TILTED_CROSS, markerSize=13, thickness=2)

            # Storing
            with self.buffer_lock:
                # Get Motor State
                with self.motor_lock:
                    self.calibration.store_latest_state(self._motor_states)
                    self.state_buffer['t-1'] = copy.deepcopy(self.state_buffer['t'])
                    self.state_buffer['t']['chest_cam_px'] = chest_cam_px
                    self.state_buffer['t']['left_eye_px'] = left_eye_px
                    self.state_buffer['t']['right_eye_px'] = right_eye_px
                    self.state_buffer['t']['theta_left_pan'] = self._motor_states[0]['angle']
                    self.state_buffer['t']['theta_right_pan'] = self._motor_states[1]['angle']
                    self.state_buffer['t']['theta_tilt'] = self._motor_states[2]['angle']
                    rospy.loginfo(str(self.state_buffer))
                
                # Calibration Algorithm
                theta_l_pan, theta_l_tilt = self.calibration.compute_left_eye_cmd(dx_l, dy_l)
                theta_r_pan, theta_r_tilt = self.calibration.compute_right_eye_cmd(dx_r, dy_r) 
                theta_tilt = self.calibration.compute_tilt_cmd(theta_l_tilt, theta_r_tilt, alpha_tilt=0.5)
                self.calibration.store_cmd(theta_l_pan, theta_r_pan, theta_tilt)

                # # Print Info
                # rospy.loginfo(str(self._motor_states))
                # rospy.loginfo(f"dx_l: {dx_l: .{4}f}")
                # rospy.loginfo(f"dy_l: {dy_l: .{4}f}")
                # rospy.loginfo(f"dx_r: {dx_r: .{4}f}")
                # rospy.loginfo(f"dy_r: {dy_r: .{4}f}")
                # rospy.loginfo(f"theta_l_pan_t: {self.calibration.buffer['t']['state']['EyeTurnLeft']['angle']: .{4}f}")
                # rospy.loginfo(f"theta_r_pan_t: {self.calibration.buffer['t']['state']['EyeTurnRight']['angle']: .{4}f}")
                # rospy.loginfo(f"theta_tilt_t: {self.calibration.buffer['t']['state']['EyesUpDown']['angle']: .{4}f}")
                # rospy.loginfo(f"theta_l_pan_cmd:: {theta_l_pan: .{4}f}")
                # rospy.loginfo(f"theta_r_pan_cmd: {theta_r_pan: .{4}f}")
                # rospy.loginfo(f"theta_tilt:_cmd: {theta_tilt: .{4}f}")
                # rospy.loginfo(f"eta_tminus1_l_pan: {self.calibration.buffer['t-1']['hidden']['EyeTurnLeft']: .{4}f}")
                # rospy.loginfo(f"eta_t_l_pan: {self.calibration.buffer['t']['hidden']['EyeTurnLeft']: .{4}f}")
                # rospy.loginfo(f"eta_tminus1_r_pan: {self.calibration.buffer['t-1']['hidden']['EyeTurnRight']: .{4}f}")
                # rospy.loginfo(f"eta_t_r_pan: {self.calibration.buffer['t']['hidden']['EyeTurnRight']: .{4}f}")
                rospy.loginfo(f"--------------")
            
            # Movement
            # theta_l_pan, theta_r_pan, theta_tilt = None, None, None
            if (theta_l_pan is not None) or (theta_r_pan is not None) or (theta_tilt is not None):
                self.move((theta_l_pan, theta_r_pan, theta_tilt))

            # Visualization
            left_img = self.ctr_cross_img(left_img, 'left_eye')
            right_img = self.ctr_cross_img(right_img, 'right_eye')
            concat_img = np.hstack((chest_img, left_img, right_img))
            
            # Resizing
            height, width = concat_img.shape[:2]
            concat_img = cv2.resize(concat_img, (round(width/2), round(height/2)))

            # Output Display 1
            self.rt_display_pub.publish(self.bridge.cv2_to_imgmsg(concat_img, encoding="bgr8"))

            # Reassigning
            self.chess_idx_tminus1 = chess_idx

            # Publish States
            json_str = json.dumps(self.state_buffer)
            message = String()
            message.data = json_str

            # Publish the message
            self.state_pub.publish(message)

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

    def ctr_cross_img(self, img, eye:str):
        """eye (str): select from ['left_eye', 'right_eye']
        """
        img = cv2.line(img, (round(self.calib_params[eye]['x_center']), 0), (round(self.calib_params[eye]['x_center']), 480), (0,255,0))
        img = cv2.line(img, (0, round(self.calib_params[eye]['y_center'])), (640, round(self.calib_params[eye]['y_center'])), (0,255,0))
        img = cv2.drawMarker(img, (round(self.calib_params[eye]['x_center']), round(self.calib_params[eye]['y_center'])), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        return img


if __name__ == '__main__':
    rospy.init_node('visuomotor')
    vismotor = VisuoMotorNode()
    rospy.spin()
