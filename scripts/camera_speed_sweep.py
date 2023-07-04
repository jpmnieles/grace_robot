import os
import sys
sys.path.append(os.getcwd())

import time
import cv2
import pandas as pd
import numpy as np

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from grace.utils import *
from datetime import datetime


class EyeCamSpeedSweep(object):
    

    def __init__(self, camera='right', show_image=True):
        self.camera = camera
        rospy.init_node("eye_cam_speed_sweep")
        self.set_show_image(show_image)
        self.bridge = CvBridge()
        self.left_eye_sub = rospy.Subscriber('/eye_camera/left_eye/image_raw', Image, self._capture_left_image)
        self.right_eye_sub = rospy.Subscriber('/eye_camera/right_eye/image_raw', Image, self._capture_right_image)
        time.sleep(1)
        self.logger = {
            'timestamp':[],
            'pixel_x': [],
            'pixel_y': [],
        }

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
    
    def main(self, motor_id, max_amplitude, trials):

        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        motor_name = capture_motor_name(motor_id)
        filename = dt_str + '_MotorSweepSpeed_' + motor_name + '_' + '%damp_'%(max_amplitude) + '%dtrials'%(trials) + ".csv"

        rate = rospy.Rate(30) # 30hz
        while not rospy.is_shutdown():
            ts = datetime.timestamp(datetime.now())
            try:
                if self.camera == 'left':
                    pixels = get_chessboard_point(img=self.left_eye_img, idx=21)
                elif self.camera == 'right':
                    pixels = get_chessboard_point(img=self.right_eye_img, idx=21)
                self.logger['timestamp'].append(ts)
                self.logger['pixel_x'].append(pixels[0])
                self.logger['pixel_y'].append(pixels[1])
                print("[%f] Pixel X: %.4f, Pixel Y: %.4f" % (ts, pixels[0], pixels[1]))
            except Exception as e:
                print(e)
            
            if self._show_image:
                cv2.imshow('Left Eye', self.left_eye_img)
                cv2.imshow('Right Eye', self.right_eye_img)
                cv2.waitKey(1)
            rate.sleep()
        
        print('-- End of Program --')
        filepath = os.path.join(os.path.abspath(""), "results", filename)
        df = pd.DataFrame(self.logger)
        df.to_csv(filepath)
        print('Data saved in:', filepath)
        
    

if __name__ == '__main__':
    eye_cam_speed_sweep = EyeCamSpeedSweep(camera='left', show_image=True)
    eye_cam_speed_sweep.main(motor_id=14, max_amplitude=22, trials=5)
