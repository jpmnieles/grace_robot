import os
import sys
import yaml
import math
import cv2


def motor_int_to_angle(motor, position, degrees):
    if degrees:
        unit = 360
    else:
        unit = math.pi
    angle = ((position-motors_dict[motor]['init'])/4096)*unit
    return angle


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


# Loading Motors Yaml File
with open(os.path.join(os.getcwd(),'config', 'head','motors.yaml'), 'r') as stream:
    try:
        head_dict = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        print(error)

with open(os.path.join(os.getcwd(),'config', 'body','motors.yaml'), 'r') as stream:
    try:
        body_dict = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        print(error)

motors_dict = {}
motors_dict.update(head_dict['motors'])
motors_dict.update(body_dict['motors'])
