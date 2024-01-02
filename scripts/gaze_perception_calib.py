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

from grace.utils import load_json
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
            self._cmd[2] += 1 # tilt eyes go up
        elif (k==115):  # letter s
            self._cmd[2] -= 1 # tilt eyes go down
        elif (k==105):  # letter i
            self._cmd[2] += 0.4395 # tilt eyes go up
        elif (k==107):  # letter k
            self._cmd[2] -= 0.4395 # tilt eyes go down
        elif (k==99):  # letter c
            key = 99  # slow confirming position
        elif (k==49):  # number 1
            key = 49  # Capture images
        elif (k==50):  # number 2
            key = 50  # Capture chessboard
        elif (k==27):
            sys.exit("Exited Progam")
        return key
    
    @property
    def cmd(self):
        return self._cmd


if __name__ == "__main__":
    # Initialization
    save_ctr = 0
    calib_params = load_json('config/calib/calib_params.json')
    m_list = [calib_params['left_eye']['slope'], calib_params['right_eye']['slope'], calib_params['tilt_eyes']['slope']]
    
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    results_dir = 'results/' + date_str 
    try:
        os.makedirs(results_dir, exist_ok = True)
        print("Directory '%s' created successfully:" %results_dir)
    except OSError as error:
        print("Directory '%s' can not be created")

    naming_dict = {0:-20, 1:-10, 2:0, 3:10, 4:20}
    save_dict = {
        -20: {
            'state': [],
            'motor_position': [],
            'eye_position': [],
            'left_img': None,
            'right_img': None,
        },
        -10: {
            'state': [],
            'motor_position': [],
            'eye_position': [],
            'left_img': None,
            'right_img': None,
        },
        0: {
            'state': [],
            'motor_position': [],
            'eye_position': [],
            'left_img': None,
            'right_img': None,
        },
        10: {
            'state': [],
            'motor_position': [],
            'eye_position': [],
            'left_img': None,
            'right_img': None,
        },
        20: {
            'state': [],
            'motor_position': [],
            'eye_position': [],
            'left_img': None,
            'right_img': None,
        },  
    }

    # Instantiation
    left_cam = LeftEyeCapture()
    right_cam = RightEyeCapture()
    client = ROSMotorClient(["EyeTurnLeft", "EyeTurnRight", "EyesUpDown"], degrees=True, debug=False)
    key_ctrl = GraceKeyboardCtrl()
    rate = rospy.Rate(30)

    client.move(key_ctrl.cmd)
    time.sleep(0.3333)

    while not rospy.is_shutdown():

        try:
            l_img = left_cam.frame
        except:
            pass
        
        try:
            r_img = right_cam.frame
        except:
            pass   

        key = key_ctrl.get_keys()
        if key == 99:  # Press letter 'c' to confirm
            print(key_ctrl.cmd)
            state = client.state
            curr_position_list = [state[idx]['angle'] for idx in range(client.num_names)]
            client.slow_move(idx=0,position=-18,step_size=0.0879,time_interval=0.015)
            client.slow_move(idx=0,position=curr_position_list[0],step_size=0.0879,time_interval=0.015)
            client.slow_move(idx=1,position=-18,step_size=0.0879,time_interval=0.015)
            client.slow_move(idx=1,position=curr_position_list[1],step_size=0.0879,time_interval=0.015)
            state = client.state
            rate.sleep()
            print(state)
        elif key == 49:  # Press number '1' to save
            angle = naming_dict[save_ctr]
            state = client.state
            l_frame = left_cam.frame
            r_frame = right_cam.frame
            print("Left Eye Camera Shape:", l_frame.shape)
            print("Right Eye Camera:", r_frame.shape)
            time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") 
            l_fn_str = os.path.join('results', date_str, str(angle)+'_left_eye.png')
            r_fn_str = os.path.join('results', date_str, str(angle)+'_right_eye.png')
            cv.imwrite(l_fn_str, l_frame)
            cv.imwrite(r_fn_str, r_frame)
            print("Saving Left Camera Image to: ", l_fn_str)
            print("Saving Right Camera Image to: ", r_fn_str)

            # JSON Save
            save_dict[angle]['state'] = state
            motor_position_list = [state[idx]['angle'] for idx in range(client.num_names)]
            save_dict[angle]['motor_position'] = motor_position_list 
            eye_position_list = [x*m_list[i] for i,x in enumerate(motor_position_list)]
            save_dict[angle]['eye_position'] = eye_position_list
            save_dict[angle]['left_img'] = l_fn_str 
            save_dict[angle]['right_img'] = r_fn_str
            save_ctr += 1

            # Program Exit
            if save_ctr == 5:
                json_fn = os.path.join('results', date_str,'data.json')
                with open(json_fn, "w") as file:
                    json.dump(save_dict, file)
                
                sys.exit("Exited Progam")
        elif key == 50:  # Press number '2' to save chessboard image
            angle = naming_dict[save_ctr]
            l_frame = left_cam.frame
            r_frame = right_cam.frame
            print("Left Eye Camera Shape:", l_frame.shape)
            print("Right Eye Camera:", r_frame.shape)
            time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") 
            l_fn_str = os.path.join('results', date_str, str(angle)+'_chess_left_eye.png')
            r_fn_str = os.path.join('results', date_str, str(angle)+'_chess_right_eye.png')
            cv.imwrite(l_fn_str, l_frame)
            cv.imwrite(r_fn_str, r_frame)
            print("Saving Left Camera Image to: ", l_fn_str)
            print("Saving Right Camera Image to: ", r_fn_str)
            save_ctr += 1

            # Program Exit
            if save_ctr == 5:
                sys.exit("Exited Progam")
        else:
            print(key_ctrl.cmd)
            client.move(key_ctrl.cmd)
            rate.sleep()
            state = client.state
            print(state)