import os
import sys
sys.path.append(os.getcwd())

import time
import json
import datetime
import numpy as np
import cv2 as cv
import getch

import rospy

from grace.camera import LeftEyeCapture, RightEyeCapture
from grace.control import ROSMotorClient


class GraceKeyboardCtrl(object):


    def __init__(self) -> None:
        self._cmd = [0,0,0]

    def reset(self):
        self._cmd = [0,0,0]
        return self._cmd

    def get_keys(self): #gets keyboard input
        key = 0
        k = ord(getch.getch()) #converts keypress to ord value
        print(k)
        if (k==97):  # letter a
            self._cmd[1] += 0.4395 # right eye go left
            print("right eye go left")
        elif (k==100):  # letter d
            self._cmd[1] -= 0.4395 # right eye go right
        elif (k==113):  # letter q
            self._cmd[1] += 0.0879 # right eye go left
            print("right eye go left")
        elif (k==101):  # letter e
            self._cmd[1] -= 0.0879 # right eye go right
        elif (k==106):  # letter j
            self._cmd[0] += 0.4395  # left eye go left
        elif (k==108):  # letter l
            self._cmd[0] -= 0.4395  # left eye go right
        elif (k==117):  # letter u
            self._cmd[0] += 0.0879 # left eye go left
        elif (k==111):  # letter o
            self._cmd[0] -= 0.0879 # left eye go right
        elif (k==119):  # letter w
            self._cmd[2] += 1 # tilt eyes go left
        elif (k==115):  # letter s
            self._cmd[2] -= 1 # tilt eyes go right
        elif (k==105):  # letter i
            self._cmd[2] += 0.4395 # tilt eyes go left
        elif (k==107):  # letter k
            self._cmd[2] -= 0.4395 # tilt eyes go right
        
        elif (k==27):
            sys.exit("Exited Progam")
        return key
    
    @property
    def cmd(self):
        return self._cmd


if __name__ == "__main__":
    # Initialization
    left_cam = LeftEyeCapture()
    right_cam = RightEyeCapture()
    client = ROSMotorClient(["EyeTurnLeft", "EyeTurnRight", "EyesUpDown"], degrees=True, debug=False)
    key_ctrl = GraceKeyboardCtrl()
    rate = rospy.Rate(30)

    client.move(key_ctrl.cmd)
    time.sleep(0.3333)

    while not rospy.is_shutdown():
        # print(client.state)
        key_ctrl.get_keys()
        print(key_ctrl.cmd)
        client.move(key_ctrl.cmd)
        rate.sleep()
    
