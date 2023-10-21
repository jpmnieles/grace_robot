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

from grace.utils import *
import threading
import copy


class BaselineCalibration(object):

    init_buffer = {
        't-1': {
            'cmd': {
                'EyeTurnLeft': 0.0,
                'EyeTurnRight': 0.0,
                'EyesUpDown': 0.0
            },
            'state': {
                'EyeTurnLeft': 0.0,
                'EyeTurnRight': 0.0,
                'EyesUpDown': 0.0
            },
        },
        't': {
            'cmd': {
                'EyeTurnLeft': 0.0,
                'EyeTurnRight': 0.0,
                'EyesUpDown': 0.0
            },
            'state': {
                'EyeTurnLeft': 0.0,
                'EyeTurnRight': 0.0,
                'EyesUpDown': 0.0
            },
        },
        't+1': {
            'cmd': {
                'EyeTurnLeft': 0.0,
                'EyeTurnRight': 0.0,
                'EyesUpDown': 0.0
            },
        }
    }


    def __init__(self, lock) -> None:
        self.lock = lock
        self.camera_mtx = load_camera_mtx()
        self.calib_params = load_json('config/calib/calib_params.json')
        self.reset_buffer()

    def reset_buffer(self):
        with self.lock:
            self.buffer = self.init_buffer.copy()

    def store_latest_state(self, latest_state):
        self.buffer['t']['cmd'] = self.buffer['t+1']['cmd']
        self.buffer['t-1'] = self.buffer['t']

        self.buffer['t']['state']['EyeTurnLeft'] = latest_state[0]
        self.buffer['t']['state']['EyeTurnRight'] = latest_state[1]
        self.buffer['t']['state']['EyesUpDown'] = latest_state[2]

    def _px_to_deg_fx(self, x, eye:str):
        """eye (str): select from ['left_eye', 'right_eye']
        """
        theta = math.atan(x/self.camera_mtx[eye]['fx'])
        theta = math.degrees(theta)
        return theta

    def _px_to_deg_fy(self, y, eye:str):
        """eye (str): select from ['left_eye', 'right_eye']
        """
        theta = math.atan(y/self.camera_mtx[eye]['fy'])
        theta = math.degrees(theta)
        return theta

    def compute_left_eye_cmd(self, dx, dy):
        theta_l_pan_tplus1 = (self.buffer['t']['state']['EyeTurnLeft']['angle'] 
                          + self._px_to_deg_fx(dx, 'left_eye')/self.calib_params['left_eye']['slope'])
        theta_l_tilt_tplus1 = (self.buffer['t']['state']['EyesUpDown']['angle'] 
                          + self._px_to_deg_fy(dy, 'left_eye')/self.calib_params['tilt_eyes']['slope'])
        return theta_l_pan_tplus1, theta_l_tilt_tplus1

    def compute_right_eye_cmd(self, dx, dy):
        theta_r_pan_tplus1 = (self.buffer['t']['state']['EyeTurnRight']['angle'] 
                          + self._px_to_deg_fx(dx, 'right_eye')/self.calib_params['right_eye']['slope'])
        theta_r_tilt_tplus1  = (self.buffer['t']['state']['EyesUpDown']['angle'] 
                          + self._px_to_deg_fy(dy, 'right_eye')/self.calib_params['tilt_eyes']['slope'])
        return theta_r_pan_tplus1, theta_r_tilt_tplus1

    def compute_tilt_cmd(self, theta_l_tilt, theta_r_tilt, alpha_tilt=0.5):
        """alpha_tilt: percentage of theta right tilt
        """
        if theta_l_tilt is None:
            theta_tilt = theta_r_tilt
        elif theta_r_tilt is None:
            theta_tilt = theta_l_tilt
        elif theta_l_tilt is None and theta_r_tilt is None:
            theta_tilt = None
        else:
            theta_tilt = (1-alpha_tilt)*theta_l_tilt + alpha_tilt*theta_r_tilt
        return theta_tilt
    
    def store_cmd(self, theta_l_pan, theta_r_pan, theta_tilt):
        if theta_l_pan is None:
            theta_l_pan = self.buffer['t']['state']['EyeTurnLeft']['angle']
        if theta_r_pan is None:
            theta_r_pan = self.buffer['t']['state']['EyeTurnRight']['angle']
        if theta_tilt is None:
            theta_tilt = self.buffer['t']['state']['EyesUpDown']['angle']

        self.buffer['t+1']['cmd']['EyeTurnLeft'] = theta_l_pan
        self.buffer['t+1']['cmd']['EyeTurnRight'] = theta_r_pan
        self.buffer['t+1']['cmd']['EyesUpDown'] = theta_tilt


class PeopleAttention(object):

    def __init__(self) -> None:
        self.camera_mtx = load_camera_mtx()
        self.person_detected = False
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.path.join(os.getcwd(),'pretrained','shape_predictor_68_face_landmarks.dat'))
        dlib.cuda.set_device(0)

    def register_imgs(self, left_img, right_img):
        self.left_img = left_img
        self.right_img  = right_img

    def detect_people(self, left_img, right_img):
        # Detection       
        self.l_gray = cv2.cvtColor(left_img.copy(), cv2.COLOR_BGR2GRAY)
        self.r_gray = cv2.cvtColor(right_img.copy(), cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        self.l_detections = self.detector(self.l_gray, 0)
        self.r_detections = self.detector(self.r_gray, 0)

        # Person detected or not
        if len(self.l_detections) > 0 or len(self.r_detections) > 0:
            self.person_detected = True
            # print(self.camera_mtx)
        else:
            self.person_detected = False

    def get_pixel_target(self, id:int, eye:str):
        """id (int): select from [0, 1, ...]
        eye (str): select from ['left_eye', 'right_eye']
        """
        if eye == 'left_eye':
            detection = self.l_detections[id]
            img = self.l_gray
        elif eye == 'right_eye':
            detection = self.r_detections[id]
            img = self.r_gray
        landmarks = self.predictor(img, detection)
        x_target = landmarks.part(30).x
        y_target = landmarks.part(30).y
        delta_x = x_target - self.camera_mtx[eye]['cx']
        delta_y =  self.camera_mtx[eye]['cy'] - y_target
        return delta_x, delta_y
    
    def visualize_target(self, delta_x, delta_y, img, id:int, eye:str):
        """eye (str): select from ['left_eye', 'right_eye']
        """
        if eye == 'left_eye':
            detection = self.l_detections[id]
        elif eye == 'right_eye':
            detection = self.r_detections[id]
        cv2.rectangle(img, (detection.left(), detection.top()), (detection.right(), detection.bottom()), (0, 0, 255), 2)
        abs_x = self.camera_mtx[eye]['cx'] + delta_x
        abs_y = self.camera_mtx[eye]['cy'] - delta_y
        disp_img = cv2.drawMarker(img, (round(abs_x),round(abs_y)), color=(255, 0, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=13, thickness=2)
        return disp_img


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

        self.attention = PeopleAttention()
        self.calibration = BaselineCalibration(self.buffer_lock)

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
        
        # rospy.loginfo(str(self._motor_states))
        # rospy.loginfo(str(temp_name_list))
        
        # elapsed_time = (curr_stamp - self.motor_stamp_tminus1).to_sec()
        # rospy.loginfo(f'FPS: {1/elapsed_time: .{2}f}')
        # self.motor_stamp_tminus1 = curr_stamp
        # rospy.loginfo('-----------')


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
                
                print(self._motor_states)
                print('dx_l:', dx_l)
                print('dy_l:', dy_l)
                print('dx_r:', dx_l)
                print('dy_l:', dy_l)
                print('theta_l_pan:', theta_l_pan)
                print('theta_r_pan:', theta_r_pan)
                print('theta_tilt:', theta_tilt)
                print('--------------')
                # rospy.loginfo(self.calibration.buffer)
            
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
        left_img_tminus1 = left_img_tminus1[round(self.camera_mtx['left_eye']['cy'])-120:round(self.camera_mtx['left_eye']['cy'])+120,
                                             round(self.camera_mtx['left_eye']['cx'])-160: round(self.camera_mtx['left_eye']['cx'])+160]
        cv2.putText(left_img_tminus1, 'Left Eye (t-1)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        right_img_tminus1 = right_img_tminus1[round(self.camera_mtx['right_eye']['cy'])-120:round(self.camera_mtx['right_eye']['cy'])+120,
                                        round(self.camera_mtx['right_eye']['cx'])-160: round(self.camera_mtx['right_eye']['cx'])+160]
        cv2.putText(right_img_tminus1, 'Right Eye (t-1)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        left_img_t = left_img_t[round(self.camera_mtx['left_eye']['cy'])-120:round(self.camera_mtx['left_eye']['cy'])+120,
                                             round(self.camera_mtx['left_eye']['cx'])-160: round(self.camera_mtx['left_eye']['cx'])+160]
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
        img = cv2.line(img, (round(self.camera_mtx[eye]['cx']), 0), (round(self.camera_mtx[eye]['cx']), 480), (0,255,0))
        img = cv2.line(img, (0, round(self.camera_mtx[eye]['cy'])), (640, round(self.camera_mtx[eye]['cy'])), (0,255,0))
        img = cv2.drawMarker(img, (round(self.camera_mtx[eye]['cx']), round(self.camera_mtx[eye]['cy'])), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        return img


if __name__ == '__main__':
    rospy.init_node('visuomotor')
    vismotor = VisuoMotorNode()
    rospy.spin()
