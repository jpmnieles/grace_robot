import time
import cv2
import numpy as np
from datetime import datetime
import copy

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class EyeCamSubscriber(object):
    

    def __init__(self, camera='left', show_image=True):
        rospy.init_node("eye_camera_subscriber")
        self.set_show_image(show_image)
        self.bridge = CvBridge()
        if camera == 'left':
            self.old_left_timestamp = datetime.timestamp(datetime.now())
            self.left_eye_sub = rospy.Subscriber('/left_eye/image_raw', Image, self._capture_left_image, queue_size=1)
        elif camera == 'right':
            self.old_right_timestamp = datetime.timestamp(datetime.now())
            self.right_eye_sub = rospy.Subscriber('/right_eye/image_raw', Image, self._capture_right_image, queue_size=1)
        self.main()

    def _capture_left_image(self, msg):
        try:
            self.left_eye_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.left_timestamp = msg.header.stamp.to_time()
            left_elapsed = (datetime.fromtimestamp(self.left_timestamp) - datetime.fromtimestamp(self.old_left_timestamp)).total_seconds()
            left_fps = 1/left_elapsed
            self.old_left_timestamp = copy.deepcopy(self.left_timestamp)
            print('<<< Left <<< ', 'Timestamp:', self.left_timestamp, 'Elapsed (sec):', left_elapsed, 'FPS:', left_fps)
            if self._show_image:
                cv2.imshow('Left Eye', self.left_eye_img)
                cv2.waitKey(1)             
        except CvBridgeError as error:
            print(error)
    
    def _capture_right_image(self, msg):
        try:
            self.right_eye_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.right_timestamp = msg.header.stamp.to_time()
            right_elapsed = (datetime.fromtimestamp(self.right_timestamp) - datetime.fromtimestamp(self.old_right_timestamp)).total_seconds()
            right_fps = 1/right_elapsed
            self.old_right_timestamp = copy.deepcopy(self.right_timestamp)
            print('>>> Right >>> ', 'Timestamp:', self.right_timestamp, 'Elapsed (sec):', right_elapsed, 'FPS:', right_fps)
            if self._show_image:
                cv2.imshow('Right Eye', self.right_eye_img)
                cv2.waitKey(1)  
        except CvBridgeError as error:
            print(error)
    
    def set_show_image(self, value:bool):
        self._show_image = value
    
    def main(self):
        rospy.spin() 
    

if __name__ == '__main__':
    eye_cam = EyeCamSubscriber(camera="right", show_image=True)
