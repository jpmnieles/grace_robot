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
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError

import cv2
import dlib

from grace.utils import *


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
        self.detect_people(self.left_img, self.right_img)

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
            print(self.camera_mtx)
        else:
            self.person_detected = False

    def process_img(self, eye:str):
        """eye (str): select from ['left_eye', 'right_eye']
        """
        if eye == 'left_eye':
            img = self.left_img
        elif eye == 'right_eye':
            img = self.right_img
        img = self.ctr_cross_img(img, eye)
        return img

    def get_pixel_target(self, id, eye:str):
        """eye (str): select from ['left_eye', 'right_eye']
        """
        if len(self.l_detections) > 0:
            landmarks = self.predictor(self.l_gray, self.l_detections[id])
            x_target = landmarks.part(30).x
            y_target = landmarks.part(30).y
            delta_x = x_target - self.camera_mtx[eye]['cx']  # 317.13846547
            delta_y =  self.camera_mtx[eye]['cy'] - y_target  # 219.22972847
    
    def _px_to_deg_fx(self, x, eye:str):
        """eye (str): select from ['left_eye', 'right_eye']
        """
        x = math.atan(x/self.camera_mtx[eye]['fx'])
        x = math.degrees(x)
        return x

    def _px_to_deg_fy(self, y, eye:str):
        """eye (str): select from ['left_eye', 'right_eye']
        """
        y = math.atan(y/self.camera_mtx[eye]['fy'])
        y = math.degrees(y)
        return y
    
    def ctr_cross_img(self, img, eye:str):
        """eye (str): select from ['left_eye', 'right_eye']
        """
        img = cv2.line(img, (round(self.camera_mtx[eye]['cx']), 0), (round(self.camera_mtx[eye]['cx']), 480), (0,255,0))
        img = cv2.line(img, (0, round(self.camera_mtx[eye]['cy'])), (640, round(self.camera_mtx[eye]['cy'])), (0,255,0))
        img = cv2.drawMarker(img, (round(self.camera_mtx[eye]['cx']), round(self.camera_mtx[eye]['cy'])), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        return img
    
    def display_target(self, delta_x, delta_y, img, eye:str):
        """eye (str): select from ['left_eye', 'right_eye']
        """
        abs_x = self.camera_mtx[eye]['cx'] + delta_x
        abs_y = self.camera_mtx[eye]['cy'] - delta_y
        disp_img = cv2.drawMarker(img, (round(abs_x),round(abs_y)), color=(255, 0, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=13, thickness=2)
        return disp_img



class VisuoMotorNode(object):
    

    def __init__(self):
        # motors=["EyeTurnLeft", "EyeTurnRight", "EyesUpDown"]
        # self.set_motor_limits(motors)
        # self._motor_state = [None]*self.num_names
        # self.motor_sub = rospy.Subscriber('/hr/actuators/motor_states', MotorStateList, self._capture_state)
        # self.motor_pub = rospy.Publisher('/hr/actuators/pose', TargetPosture, queue_size=5)
        self.bridge = CvBridge()
        self.left_eye_sub = message_filters.Subscriber("/left_eye/image_raw", Image)
        self.right_eye_sub = message_filters.Subscriber("/right_eye/image_raw", Image)
        self.ats = message_filters.ApproximateTimeSynchronizer([self.left_eye_sub, self.right_eye_sub], queue_size=1, slop=0.015)
        self.ats.registerCallback(self._eye_imgs_callback)
        # self.motors_ready_sub = rospy.Subscriber('/motors_ready', Bool, self._motors_ready_callback)
        self.attention = PeopleAttention()
        # self.calibration = BaselineCalibration()
        self.display_l_img_pub = rospy.Publisher('/left_eye/image_processed', Image, queue_size=1)
        self.display_r_img_pub = rospy.Publisher('/right_eye/image_processed', Image, queue_size=1)
        self.motors_ready = True
    
    # def _motors_ready_callback(self, msg):
    #     self.motors_ready = msg.data

    def _eye_imgs_callback(self, left_img_msg, right_img_msg):
        self.left_img = self.bridge.imgmsg_to_cv2(left_img_msg, "bgr8")
        self.right_img = self.bridge.imgmsg_to_cv2(right_img_msg, "bgr8")
        # print(left_img_msg.header, right_img_msg.header)
        
        if self.motors_ready:
            self.attention.register_imgs(self.left_img, self.right_img)
            if self.attention.person_detected:
                id = 0  # self.attention.get_target()
        #     #     dx_l, dy_l = self.attention.get_pixel_coord(id, 'left')
        #     #     dx_r, dy_r = self.attention.get_pixel_coord(id, 'right')
                self.left_img = self.attention.process_img('left')
                self.right_img = self.attention.process_img('right')
                
        #     #     theta_l_pan, theta_l_tilt = self.calibration.compute_left_img(dx_l, dy_1)
        #     #     theta_r_pan, theta_r_tilt = self.calibration.compute_right_img(dx_l, dy_r)
        #     #     theta_tilt = self.calibration.compute_tilt(theta_l_tilt, theta_r_tilt)
        #     #     delta_theta_max = self.calibration.store_command(theta_l_pan, theta_r_pan, theta_tilt)
        #     #     self.motor_pub.publish(theta_l_pan, theta_r_pan, theta_tilt, delta_theta_max)
        #     pass

        #TODO: Add fixation cross here
        self.display_l_img_pub.publish(self.bridge.cv2_to_imgmsg(self.left_img, encoding="bgr8"))
        self.display_r_img_pub.publish(self.bridge.cv2_to_imgmsg(self.right_img, encoding="bgr8"))


if __name__ == '__main__':
    rospy.init_node('visuomotor')
    vismotor = VisuoMotorNode()
    rospy.spin()
