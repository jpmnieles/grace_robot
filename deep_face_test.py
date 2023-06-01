
import os
import sys
sys.path.append(os.path.abspath('..'))

import pickle
import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import copy
import random
import pandas as pd
from sklearn.metrics import mean_squared_error

import datetime
import time

import dlib
import cv2

from grace.control import ROSMotorClient
from grace.camera import LeftEyeCapture, RightEyeCapture


# Experiment Helper Functions
def get_chessboard_points(img):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (9,6),None)
    if ret == True:
        r_corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        s_corners = r_corners.squeeze()
    return s_corners


def get_center_chessboard_point(img):
    corners = get_chessboard_points(img)
    return corners[21].tolist()


def get_chessboard_point(img, i):
    corners = get_chessboard_points(img)
    return corners[i].tolist()

def generate_triangle_wave(init_amp, min_amp, max_amp, step_size, num_cycles, include_init):
    """Generates a triangular wave with positive and negative peaks

    Args:
        init_amp (float): initial amplitude
        min_amp (float): minimum amplitude peak
        max_amp (float): maximum amplitude peak
        step_size (float): amplitude resolution
        num_cycles (int): number of cycles of the triangular wave
        include_init (bool_): will or will not include initial amplitude in the end of waveform

    Returns:
        numpy array:
    """
    int_init_amp = round(init_amp/step_size)
    int_max_amp = round(max_amp/step_size)
    int_min_amp = round(min_amp/step_size)
    int_sweep = list(range(int_init_amp, int_max_amp+1)) + list(range(int_max_amp-1, int_min_amp-1, -1)) + list(range(int_min_amp+1, int_init_amp))
    single_sweep = [step_size*x for x in int_sweep]

    triangle_wave = []
    for _ in range(num_cycles):
        triangle_wave += single_sweep
    if include_init:
        triangle_wave.append(int_init_amp*step_size)
    return np.array(triangle_wave)

def px_to_deg_fx(x):
    x = math.atan(x/569.4456315)
    x = math.degrees(x)
    return x

def px_to_deg_fy(x):
    x = math.atan(x/571.54490033)
    x = math.degrees(x)
    return x

def save_pickle_data(data, camera: str, name: str):
    # Making Directory
    filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + camera + "_" + name
    filepath = os.path.join(os.path.abspath(".."), "results", filename)

    # Saving to Pickle File
    with open(filepath + ".pickle", 'wb') as file:
        pickle.dump(data, file)
    print('Data saved in:', filepath + ".pickle")
    return filepath + ".pickle"

def ctr_cross_img(img):
    img = cv.line(img, (315, 0), (315, 480), (0,255,0))
    img = cv.line(img, (0, 202), (640, 202), (0,255,0))
    img = cv.drawMarker(img, (315,202), color=(0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=15, thickness=2)
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
    disp_img = cv.drawMarker(img, (round(abs_x),round(abs_y)), color=(255, 0, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=13, thickness=2)
    return disp_img


def main():

    # Instantiation
    grace = ROSMotorClient(["EyeTurnLeft", "EyesUpDown"], degrees=True, debug=False)
    left_cam = LeftEyeCapture() 
   
    # Load the pre-trained face detection and landmark detection models
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(os.path.join(os.getcwd(),'pretrained','shape_predictor_68_face_landmarks.dat'))

    # Enable CUDA for GPU acceleration (if available)
    dlib.cuda.set_device(0)

    # Checking Motor State
    grace_state = grace.state
    theta_p = grace_state[0]['angle']
    theta_t = grace_state[1]['angle']

    while (1):

        # Capture a frame
        base_img = left_cam.frame
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
            cmd_p = theta_p + px_to_deg_fx(delta_x)/1.6328
            cmd_t = theta_t + px_to_deg_fy(delta_y)/0.3910

            start_state = grace.state
            end_state = grace.move([cmd_p, cmd_t])
            elapsed_time = grace.get_elapsed_time(start_state, end_state)
            print(f"Move Elapsed Time: {elapsed_time:.8f} sec")
            theta_p = end_state[0]['angle']
            theta_t = end_state[1]['angle']

        # Display the output image
        cv2.imshow('Output', img)
        key = cv2.waitKey(1)

        if key == 27:  # Esc
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()