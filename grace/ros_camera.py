import time
import cv2
import numpy as np

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class EyeCamSubscriber(object):
    

    def __init__(self, show_image=True):
        rospy.init_node("eye_camera_subscriber")
        self.set_show_image(show_image)
        self.bridge = CvBridge()
        self.left_eye_sub = rospy.Subscriber('/eye_camera/left_eye/image_raw', Image, self._capture_left_image)
        self.right_eye_sub = rospy.Subscriber('/eye_camera/right_eye/image_raw', Image, self._capture_right_image)
        time.sleep(1)
        self.main()

    def _capture_left_image(self, msg):
        try:
            self.left_eye_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as error:
            print(error)
    
    def _capture_right_image(self, msg):
        try:
            self.right_eye_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as error:
            print(error)
    
    def set_show_image(self, value:bool):
        self._show_image = value
    
    def main(self):
        rate = rospy.Rate(30) # 30hz
        while not rospy.is_shutdown():
            if self._show_image:
                cv2.imshow('Left Eye', self.left_eye_img)
                cv2.imshow('Right Eye', self.right_eye_img)
                cv2.waitKey(1)
            rate.sleep()
    

if __name__ == '__main__':
    eye_cam = EyeCamSubscriber(show_image=True)
