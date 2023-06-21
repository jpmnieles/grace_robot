import os
import math
import time
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import dlib
import cv2

from grace.control import ROSMotorClient
from aec.baseline import PanBacklash, TiltPolicy

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


# Experiment Helper Functions
def get_chessboard_points(img):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    if ret == True:
        r_corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        s_corners = r_corners.squeeze()
    return s_corners


def get_center_chessboard_point(img):
    corners = get_chessboard_points(img)
    return corners[21].tolist()


def get_chessboard_point(img, i):
    corners = get_chessboard_points(img)
    return corners[i].tolist()

def px_to_deg_fx(x):
    x = math.atan(x/569.4456315)
    x = math.degrees(x)
    return x

def px_to_deg_fy(x):
    x = math.atan(x/571.54490033)
    x = math.degrees(x)
    return x

def ctr_cross_img(img):
    img = cv2.line(img, (315, 0), (315, 480), (0,255,0))
    img = cv2.line(img, (0, 202), (640, 202), (0,255,0))
    img = cv2.drawMarker(img, (315,202), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
    return img

def get_chess_target(idx, img):
    target = get_chessboard_point(img, idx)
    delta_x = target[0]-314.69441889
    delta_x_deg = px_to_deg_fx(delta_x)
    delta_y =  201.68845842 - target[1]
    delta_y_deg = px_to_deg_fy(delta_y)
    
    print("Delta x (px)=", delta_x)
    print("Delta x (deg)=", delta_x_deg)
    print("Delta y (px)=", delta_y)
    print("Delta y (deg)=", delta_y_deg)
    
    return delta_x, delta_y

def display_target(delta_x, delta_y, img):
    abs_x = 314.69441889 + delta_x
    abs_y = 201.68845842 - delta_y
    disp_img = cv2.drawMarker(img, (round(abs_x),round(abs_y)), color=(255, 0, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=13, thickness=2)
    return disp_img


def main(enable_logging=True, enable_baseline_policy=True):

    # Instantiation
    grace = ROSMotorClient(["EyeTurnLeft", "EyesUpDown"], degrees=True, debug=False)
    if enable_baseline_policy:
        pan_backlash = PanBacklash()
        tilt_policy = TiltPolicy()
    rate = rospy.Rate(30) # 30 Hz
    logger = {'timestamp':[], 'theta_p': [], 'theta_t':[], 'delta_x':[], 'delta_y':[], 'cmd_p':[], 'cmd_t':[], 'elapsed_time':[]}
   
    # Load the pre-trained face detection and landmark detection models
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(os.path.join(os.getcwd(),'pretrained','shape_predictor_68_face_landmarks.dat'))

    # Enable CUDA for GPU acceleration (if available)
    dlib.cuda.set_device(0)

    # Checking Motor State
    grace_state = grace.state
    theta_p = grace_state[0]['angle']
    theta_t = grace_state[1]['angle']

    # Initial Log
    print(f"[{datetime.timestamp(datetime.now())}] Running")

    while (1):

        # Capture a frame
        base_img = grace.left_eye_img
        img = base_img.copy()
          
        # Detection       
        gray = cv2.cvtColor(base_img.copy(), cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        detections = detector(gray, 0)

        # Loop over each detected face
        for detection in detections:
            # Get the facial landmarks for the detected face
            landmarks = predictor(gray, detection)

            # Draw markers on the detected face
            for i in range(68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

            # Draw a bounding box around the detected face
            x1 = detection.left()
            y1 = detection.top()
            x2 = detection.right()
            y2 = detection.bottom()
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Identifying Target
        if len(detections) > 0:
            landmarks = predictor(gray, detections[0])
            x_target = landmarks.part(30).x
            y_target = landmarks.part(30).y
            delta_x = x_target-317.13846547
            delta_y =  219.22972847 - y_target

            img = display_target(delta_x, delta_y,img)
            img = ctr_cross_img(img)

            # Command Robot
            if enable_baseline_policy:
                cmd_p, pos_p = pan_backlash.calc_cmd(delta_x, theta_p)
                cmd_t, pos_t = tilt_policy.calc_cmd(delta_y, theta_t)
            else:
                cmd_p = theta_p + px_to_deg_fx(delta_x)/1.6328
                cmd_t = theta_t + px_to_deg_fy(delta_y)/0.3910
            
            start_state = grace.state
            end_state = grace.move([cmd_p, cmd_t])
            elapsed_time = grace.get_elapsed_time(start_state, end_state)
            print(f"Move Elapsed Time: {elapsed_time:.8f} sec")
            theta_p = end_state[0]['angle']
            theta_t = end_state[1]['angle']

            # Logging
            if enable_logging:
                logger = {'timestamp':[], 'theta_p': [], 'theta_t':[], 'delta_x':[], 'delta_y':[], 'cmd_p':[], 'cmd_t':[], 'elapsed_time':[]}
                logger['timestamp'].append(start_state[0]["timestamp"])
                logger['theta_p'].append(theta_p)
                logger['theta_t'].append(theta_t)
                logger['delta_x'].append(delta_x)
                logger['delta_y'].append(delta_y)
                logger['cmd_p'].append(cmd_p)
                logger['cmd_t'].append(cmd_t)
                logger['elapsed_time'].append(elapsed_time)
            print('[timestamp]', start_state[0]["timestamp"],'theta_p:', theta_p, 'theta_t:', theta_t,
                'delta_x:', delta_x, 'delta_y:', delta_y,'cmd_p:', cmd_p,'cmd_t:', cmd_t, 'elapsed_time:',elapsed_time)

        # Display the output image
        # cv2.imshow('Output', img)
        key = cv2.waitKey(1)

        if key == 27:  # Esc
            if enable_logging:
                if enable_baseline_policy:
                    filename = datetime.now().strftime("%d%m%Y_%H%M%S") + "_baseline_policy" + ".csv"
                else:
                    filename = datetime.now().strftime("%d%m%Y_%H%M%S") + ".csv"
                df = pd.DataFrame(logger)
                df.to_csv(filename)
            break
        rate.sleep()

if __name__ == "__main__":
    main(enable_logging=True, enable_baseline_policy=True)