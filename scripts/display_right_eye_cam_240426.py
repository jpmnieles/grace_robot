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


class RightEyeCam(object):

    
    def __init__(self) -> None:
        self.camera_mtx = load_json("config/camera/camera_mtx.json")
        self.camera_mtx_params = load_camera_mtx()
        self.calib_params = load_json('config/calib/calib_params.json')
        
        self.attention = ArucoAttention()

        self.bridge = CvBridge()
        self.right_eye_sub = rospy.Subscriber("/right_eye/image_raw", Image, self.eye_img_callback)
        
        rospy.loginfo('Running')

    def eye_img_callback(self, msg):
        eye = 'right_eye'
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        img = cv2.putText(img, eye, (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,255,0), 2, cv2.LINE_AA)
        
        _, img = self.attention.process_img(copy.deepcopy(img),
                                                      np.array(self.camera_mtx[eye]['camera_matrix']),
                                                      np.array(self.camera_mtx[eye]['distortion_coefficients']))

        img = cv2.line(img, (round(self.calib_params[eye]['x_center']), 0), (round(self.calib_params[eye]['x_center']), 480), (0,255,0))
        img = cv2.line(img, (0, round(self.calib_params[eye]['y_center'])), (640, round(self.calib_params[eye]['y_center'])), (0,255,0))
        img = cv2.drawMarker(img, (round(self.calib_params[eye]['x_center']), round(self.calib_params[eye]['y_center'])), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        
        try:
            cv2.imshow("Right Camera", img)
        except Exception as E:
            print(E)
        key = cv2.waitKey(1)


if __name__ == '__main__':
    rospy.init_node('display_right_eye')
    right_eye_cam = RightEyeCam()
    rospy.spin()