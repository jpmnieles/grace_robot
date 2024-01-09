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
from std_msgs.msg import Float32
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
from grace.attention import PeopleAttention


class VisuoMotorNode(object):
    
    camera_buffer = {
        't-1': {
            'left_eye': np.zeros((480,640,3), dtype=np.uint8),
            'right_eye': np.zeros((480,640,3), dtype=np.uint8),
        },
        't': {
            'left_eye': np.zeros((480,640,3), dtype=np.uint8),
            'right_eye': np.zeros((480,640,3), dtype=np.uint8)
        },
    }


    def __init__(self, motors=["EyeTurnLeft", "EyeTurnRight", "EyesUpDown"], degrees=True):
        self.motor_lock = threading.Lock()
        self.buffer_lock = threading.Lock()
        self.motors = motors
        self.degrees = degrees
        self._set_motor_limits(motors)
        self.calib_params = load_json('config/calib/calib_params.json')

        self.attention = PeopleAttention()
        self.calibration = BaselineCalibration(self.buffer_lock)
        self.calibration.toggle_backlash(True)
        self.frame_stamp_tminus1 = rospy.Time.now()
        self.motor_stamp_tminus1 = rospy.Time.now()

        self.motor_pub = rospy.Publisher('/hr/actuators/pose', TargetPosture, queue_size=1)
        self.camera_mtx = load_camera_mtx()
        self.bridge = CvBridge()
        time.sleep(1)

        # self._motor_state = [None]*self.num_names
        self.motor_sub = rospy.Subscriber('/hr/actuators/motor_states', MotorStateList, self._capture_state)
        self.left_eye_sub = message_filters.Subscriber("/left_eye/image_raw", Image)
        self.right_eye_sub = message_filters.Subscriber("/right_eye/image_raw", Image)  # TODO: change to right eye when there is better camera
        self.ats = message_filters.ApproximateTimeSynchronizer([self.left_eye_sub, self.right_eye_sub], queue_size=1, slop=0.015)
        self.ats.registerCallback(self.eye_imgs_callback)

        self.rt_display_pub = rospy.Publisher('/output_display1', Image, queue_size=1)
        self.motor_display_pub = rospy.Publisher('/output_display2', Image, queue_size=1)

        self.disp_img = np.zeros((480,640,3), dtype=np.uint8)



    def _capture_state(self, msg):
        """Callback for capturing motor state
        """
        curr_stamp = rospy.Time.now()
        eye_motors_list = []
        temp_name_list= []
        self._msg = msg
        for idx, x in enumerate(msg.motor_states):
            # temp_name_list.append(x.name)
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


    def eye_imgs_callback(self, left_img_msg, right_img_msg):
        # Motor Trigger Sync (3.33 FPS or 299.99 ms)
        max_stamp = max(left_img_msg.header.stamp, right_img_msg.header.stamp)
        elapsed_time = (max_stamp - self.frame_stamp_tminus1).to_sec()
        if elapsed_time > 283e-3:
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
            # print(left_img_msg.header, right_img_msg.header)

            # Face Detection
            self.attention.register_imgs(self.left_img, self.right_img)
            self.attention.detect_people(self.left_img, self.right_img)


            # Attention
            id = 0  # Person ID
            if len(self.attention.l_detections) > 0:
                dx_l, dy_l = self.attention.get_pixel_target(id, 'left_eye')
                self.left_img = self.attention.visualize_target(dx_l, dy_l, self.left_img, id, 'left_eye')

            if len(self.attention.r_detections) > 0:
                dx_r, dy_r = self.attention.get_pixel_target(id, 'right_eye')
                self.right_img = self.attention.visualize_target(dx_r, dy_r, self.right_img, id, 'right_eye')

            # Snapshot
            self.camera_buffer['t-1']['left_eye'] = copy.deepcopy(self.camera_buffer['t']['left_eye'])
            self.camera_buffer['t']['left_eye'] = copy.deepcopy(self.left_img)
            self.camera_buffer['t-1']['right_eye'] = copy.deepcopy(self.camera_buffer['t']['right_eye'])
            self.camera_buffer['t']['right_eye'] = copy.deepcopy(self.right_img)

            # Storing
            with self.buffer_lock:
                # Get Motor State
                with self.motor_lock:
                    self.calibration.store_latest_state(self._motor_states)
                
                # Calibration Algorithm
                theta_l_pan, theta_l_tilt = self.calibration.compute_left_eye_cmd(dx_l, dy_l)
                theta_r_pan, theta_r_tilt = self.calibration.compute_right_eye_cmd(dx_r, dy_r) 
                theta_tilt = self.calibration.compute_tilt_cmd(theta_l_tilt, theta_r_tilt, alpha_tilt=0.5)
                self.calibration.store_cmd(theta_l_pan, theta_r_pan, theta_tilt)
                
                rospy.loginfo(str(self._motor_states))
                rospy.loginfo(f"dx_l: {dx_l: .{4}f}")
                rospy.loginfo(f"dy_l: {dy_l: .{4}f}")
                rospy.loginfo(f"dx_r: {dx_r: .{4}f}")
                rospy.loginfo(f"dy_r: {dy_r: .{4}f}")
                rospy.loginfo(f"theta_l_pan_t: {self.calibration.buffer['t']['state']['EyeTurnLeft']['angle']: .{4}f}")
                rospy.loginfo(f"theta_r_pan_t: {self.calibration.buffer['t']['state']['EyeTurnRight']['angle']: .{4}f}")
                rospy.loginfo(f"theta_tilt_t: {self.calibration.buffer['t']['state']['EyesUpDown']['angle']: .{4}f}")
                rospy.loginfo(f"theta_l_pan_cmd:: {theta_l_pan: .{4}f}")
                rospy.loginfo(f"theta_r_pan_cmd: {theta_r_pan: .{4}f}")
                rospy.loginfo(f"theta_tilt:_cmd: {theta_tilt: .{4}f}")
                rospy.loginfo(f"eta_tminus1_l_pan: {self.calibration.buffer['t-1']['hidden']['EyeTurnLeft']: .{4}f}")
                rospy.loginfo(f"eta_t_l_pan: {self.calibration.buffer['t']['hidden']['EyeTurnLeft']: .{4}f}")
                rospy.loginfo(f"eta_tminus1_r_pan: {self.calibration.buffer['t-1']['hidden']['EyeTurnRight']: .{4}f}")
                rospy.loginfo(f"eta_t_r_pan: {self.calibration.buffer['t']['hidden']['EyeTurnRight']: .{4}f}")
                rospy.loginfo(f"--------------")
            
            # Movement
            self.move((theta_l_pan, theta_r_pan, theta_tilt))

            # Visualization
            self.left_img = self.ctr_cross_img(self.left_img, 'left_eye')
            self.right_img = self.ctr_cross_img(self.right_img, 'right_eye')
            concat_img = np.hstack((self.left_img, self.right_img))

            if len(self.attention.l_detections) > 0 and len(self.attention.l_detections) > 0:
                self.disp_img = self.visualize_targets()

            # Output Display 1
            self.rt_display_pub.publish(self.bridge.cv2_to_imgmsg(concat_img, encoding="bgr8"))

            # Output Display 2                
            self.motor_display_pub.publish(self.bridge.cv2_to_imgmsg(self.disp_img, encoding="bgr8"))

    def visualize_targets(self):
        # Center Marker
        left_img_tminus1 = self.ctr_cross_img(copy.deepcopy(self.camera_buffer['t-1']['left_eye']), 'left_eye')
        right_img_tminus1 = self.ctr_cross_img(copy.deepcopy(self.camera_buffer['t-1']['right_eye']), 'right_eye')
        left_img_t = self.ctr_cross_img(copy.deepcopy(self.camera_buffer['t']['left_eye']), 'left_eye')
        right_img_t = self.ctr_cross_img(copy.deepcopy(self.camera_buffer['t']['right_eye']), 'right_eye')

        # Cropping
        left_img_tminus1 = left_img_tminus1[round(self.calib_params['left_eye']['y_center'])-120:round(self.calib_params['left_eye']['y_center'])+120,
                                             round(self.calib_params['left_eye']['x_center'])-160: round(self.calib_params['left_eye']['x_center'])+160]
        cv2.putText(left_img_tminus1, 'Left Eye (t-1)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        right_img_tminus1 = right_img_tminus1[round(self.camera_mtx['right_eye']['cy'])-120:round(self.camera_mtx['right_eye']['cy'])+120,
                                        round(self.camera_mtx['right_eye']['cx'])-160: round(self.camera_mtx['right_eye']['cx'])+160]
        cv2.putText(right_img_tminus1, 'Right Eye (t-1)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        left_img_t = left_img_t[round(self.calib_params['left_eye']['y_center'])-120:round(self.calib_params['left_eye']['y_center'])+120,
                                             round(self.calib_params['left_eye']['x_center'])-160: round(self.calib_params['left_eye']['x_center'])+160]
        cv2.putText(left_img_t, 'Left Eye (t)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        right_img_t= right_img_t[round(self.camera_mtx['right_eye']['cy'])-120:round(self.camera_mtx['right_eye']['cy'])+120,
                                        round(self.camera_mtx['right_eye']['cx'])-160: round(self.camera_mtx['right_eye']['cx'])+160]
        cv2.putText(right_img_t, 'Right Eye (t)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Concatenation
        before_imgs = np.hstack((left_img_tminus1, right_img_tminus1))
        after_imgs = np.hstack((left_img_t, right_img_t))
        disp_img = np.vstack((before_imgs, after_imgs))

        return disp_img

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
