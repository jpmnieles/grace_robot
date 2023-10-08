import os
import sys
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
        # self.attention = PeopleAttention()
        # self.calibration = BaselineCalibration()
        self.display_l_img_pub = rospy.Publisher('/left_eye/image_processed', Image, queue_size=1)
        self.display_r_img_pub = rospy.Publisher('/right_eye/image_processed', Image, queue_size=1)
        self.motors_ready = False
    
    # def _motors_ready_callback(self, msg):
    #     self.motors_ready = msg.data

    def _eye_imgs_callback(self, left_img_msg, right_img_msg):
        self.left_img = self.bridge.imgmsg_to_cv2(left_img_msg, "bgr8")
        self.right_img = self.bridge.imgmsg_to_cv2(right_img_msg, "bgr8")
        # print(left_img_msg.header, right_img_msg.header)
        
        # if self.motors_ready:
        #     # self.attention.register_imgs(self.left_img, self.right_img)
        #     # if self.attention.people_detected:
        #     #     id = self.attention.get_target()
        #     #     x_l, y_l = self.attention.get_pixel_coord(id, 'left')
        #     #     x_r, y_r = self.attention.get_pixel_coord(id, 'right')
        #     #     self.left_img = self.attention.process_img('left')
        #     #     self.right_img = self.attention.process_img('right')
                
        #     #     theta_l_pan, theta_l_tilt = self.calibration.compute_left_img(x_l, x_r)
        #     #     theta_r_pan, theta_r_tilt = self.calibration.compute_right_img(x_l, x_r)
        #     #     theta_tilt = self.calibration.compute_tilt(theta_l_tilt, theta_r_tilt)
        #     #     delta_theta_max = self.calibration.store_command(theta_l_pan, theta_r_pan, theta_tilt)
        #     #     self.motor_pub.publish(theta_l_pan, theta_r_pan, theta_tilt, delta_theta_max)
        #     pass

        self.display_l_img_pub.publish(self.bridge.cv2_to_imgmsg(self.left_img, encoding="bgr8"))
        self.display_r_img_pub.publish(self.bridge.cv2_to_imgmsg(self.right_img, encoding="bgr8"))


if __name__ == '__main__':
    rospy.init_node('visuomotor')
    vismotor = VisuoMotorNode()
    rospy.spin()
