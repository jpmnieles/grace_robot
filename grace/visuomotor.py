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


    def __init__(self) -> None:
        self.camera_mtx = load_camera_mtx()
        self.calib_params = load_json('config/calib/calib_params.json')
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = self.init_buffer.copy()

    def store_latest_state(self, latest_state):
        self.buffer['t']['cmd'] = self.buffer['t+1']['cmd']
        self.buffer['t-1'] = self.buffer['t']
        self.buffer['t']['state'] = latest_state

    def _px_to_deg_fx(self, x, eye:str):
        """eye (str): select from ['left_eye', 'right_eye']
        """
        theta = math.atan(x/self.camera_mtx[eye]['fx'])
        theta = math.degrees(x)
        return theta

    def _px_to_deg_fy(self, y, eye:str):
        """eye (str): select from ['left_eye', 'right_eye']
        """
        theta = math.atan(y/self.camera_mtx[eye]['fy'])
        theta = math.degrees(y)
        return theta

    def compute_left_eye_cmd(self, dx, dy):
        theta_l_pan_tplus1 = (self.buffer['t']['state']['EyeTurnLeft'] 
                          + self._px_to_deg_fx(dx, 'left_eye')/self.calib_params['left_eye']['slope'])
        theta_l_tilt_tplus1 = (self.buffer['t']['state']['EyesUpDown'] 
                          + self._px_to_deg_fy(dy, 'left_eye')/self.calib_params['tilt_eyes']['slope'])
        return theta_l_pan_tplus1, theta_l_tilt_tplus1

    def compute_right_eye_cmd(self, dx, dy):
        theta_r_pan_tplus1 = (self.buffer['t']['state']['EyeTurnRight'] 
                          + self._px_to_deg_fx(dx, 'right_eye')/self.calib_params['right_eye']['slope'])
        theta_r_tilt_tplus1  = (self.buffer['t']['state']['EyesUpDown'] 
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
            theta_l_pan = self.buffer['t']['state']['EyeTurnLeft']
        if theta_r_pan is None:
            theta_r_pan = self.buffer['t']['state']['EyeTurnRight']
        if theta_tilt is None:
            theta_tilt = self.buffer['t']['state']['EyesUpDown']

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
            'left_eye': None,
            'right_eye': None
        },
        't': {
            'left_eye': None,
            'right_eye': None
        },
    }


    def __init__(self, motors=["EyeTurnLeft", "EyeTurnRight", "EyesUpDown"], degrees=True):
        self.motors = motors
        self.degrees = degrees
        self._set_motor_limits(motors)
        # self._motor_state = [None]*self.num_names
        # self.motor_sub = rospy.Subscriber('/hr/actuators/motor_states', MotorStateList, self._capture_state)
        self.motor_pub = rospy.Publisher('/hr/actuators/pose', TargetPosture, queue_size=1)
        self.camera_mtx = load_camera_mtx()
        self.bridge = CvBridge()
        # self.left_eye_motor_sub = message_filters.Subscriber("/motor_state/left_eye_motor", Float32)
        # self.right_eye_motor_sub = message_filters.Subscriber("/motor_state/right_eye_motor", Float32)
        # self.tilt_eye_motors_sub = message_filters.Subscriber("/motor_state/tilt_eyes_motor", Float32)
        # self.ats_motors = message_filters.ApproximateTimeSynchronizer([self.left_eye_motor_sub, 
        #                                                                self.right_eye_motor_sub,
        #                                                                self.tilt_eye_motors_sub], 
        #                                                                queue_size=1, slop=0.01,
        #                                                                allow_headerless=True)
        # self.ats_motors.registerCallback(self.motor_states_callback)
        self.motors_sub = rospy.Subscriber('/motor_states', Float32, self.motor_states_callback)
        time.sleep(1)
        self.left_eye_sub = message_filters.Subscriber("/left_eye/image_raw", Image)
        self.right_eye_sub = message_filters.Subscriber("/left_eye/image_raw", Image)  # TODO: change to right eye when there is better camera
        self.ats = message_filters.ApproximateTimeSynchronizer([self.left_eye_sub, self.right_eye_sub], queue_size=1, slop=0.015)
        self.ats.registerCallback(self.eye_imgs_callback)

        self.attention = PeopleAttention()
        self.calibration = BaselineCalibration()
        self.rt_l_display_pub = rospy.Publisher('/left_eye/image_processed', Image, queue_size=1)
        self.rt_r_display_pub = rospy.Publisher('/right_eye/image_processed', Image, queue_size=1)
        self.motor_display_pub = rospy.Publisher('/eyes/image_processed', Image, queue_size=1)
        self.frame_trigger = 6  # 5.05 fps = 1/(6*0.033) ; given frame_count = 6
        self.frame_ctr = 0

    def eye_imgs_callback(self, left_img_msg, right_img_msg):
        # Initialization
        dx_l, dy_l, dx_r, dy_r = 0, 0, 0, 0
        theta_l_pan, theta_r_pan = None, None
        theta_l_tilt, theta_r_tilt, theta_tilt = None, None, None
        
        # Frame Counter
        self.frame_ctr += 1

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
        
        # Output Display 1
        self.rt_l_display_pub.publish(self.bridge.cv2_to_imgmsg(self.left_img, encoding="bgr8"))
        self.rt_r_display_pub.publish(self.bridge.cv2_to_imgmsg(self.right_img, encoding="bgr8"))
        
        # Motor Trigger
        if self.frame_ctr == self.frame_trigger:
            # Get Motor State
            self.calibration.store_latest_state(self._motor_state)
            
            # Calibration Algorithm
            theta_l_pan, theta_l_tilt = self.calibration.compute_left_eye_cmd(dx_l, dy_l)
            theta_r_pan, theta_r_tilt = self.calibration.compute_right_eye_cmd(dx_r, dy_r) 
            theta_tilt = self.calibration.compute_tilt_cmd(theta_l_tilt, theta_r_tilt, alpha_tilt=0.5)
            self.calibration.store_cmd(theta_l_pan, theta_r_pan, theta_tilt)
            self.move((theta_l_pan, theta_r_pan, theta_tilt))

            # Visualization
            self.left_img = self.ctr_cross_img(self.left_img, 'left_eye')
            self.right_img = self.ctr_cross_img(self.right_img, 'right_eye')
            concat_img = np.hstack((self.left_img, self.right_img))

            # Output Display 2
            self.motor_display_pub.publish(self.bridge.cv2_to_imgmsg(concat_img, encoding="bgr8"))

            # Reset Frame Counter
            self.frame_ctr = 0

    def motor_states_callback(self, msg):
        self._motor_state = {
            'EyeTurnLeft': msg.data,
            'EyeTurnRight': msg.data + random.randint(-1,1) + random.random(),
            'EyesUpDown': msg.data + random.randint(-4,4) + random.random()
        }

    def _convert_to_angle(self, motor, position):
        if self.degrees:
            unit = 360
        else:
            unit = math.pi
        angle = ((position-self._motor_limits[motor]['int_init'])/4096)*unit
        return angle

    def _convert_to_motor_int(self, motor, angle):
        if self.degrees:
            unit = 360
        else:
            unit = math.pi
        angle = round((angle/unit)*4096 + self._motor_limits[motor]['int_init'])
        return angle
        
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
