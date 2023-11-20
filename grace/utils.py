import os
import sys
import yaml
import math
import cv2
import json


def load_json(filename: str):
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path by joining the current directory and relative path
    absolute_path = os.path.join(current_dir, '..', filename)

    # Load the JSON data
    with open(absolute_path, 'r') as file:
        json_data = json.load(file)

    return json_data

def load_camera_mtx():
    camera_mtx = load_json("config/camera/camera_mtx.json")

    camera_mtx['left_eye']['fx'] = camera_mtx['left_eye']['camera_matrix'][0][0]
    camera_mtx['left_eye']['cx'] = camera_mtx['left_eye']['camera_matrix'][0][2]
    camera_mtx['left_eye']['fy'] = camera_mtx['left_eye']['camera_matrix'][1][1]
    camera_mtx['left_eye']['cy'] = camera_mtx['left_eye']['camera_matrix'][1][2]

    camera_mtx['right_eye']['fx'] = camera_mtx['right_eye']['camera_matrix'][0][0]
    camera_mtx['right_eye']['cx'] = camera_mtx['right_eye']['camera_matrix'][0][2]
    camera_mtx['right_eye']['fy'] = camera_mtx['right_eye']['camera_matrix'][1][1]
    camera_mtx['right_eye']['cy'] = camera_mtx['right_eye']['camera_matrix'][1][2]

    return camera_mtx


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
    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria).squeeze()
    return  ret, corners


def get_chessboard_point(img, idx):
    ret, corner = get_chessboard_points(img)
    if ret:
        corner = corner[idx].tolist()
    return ret, corner


def capture_motor_name(motor_id):
    for name in motors_dict.keys():
        if motors_dict[name]["motor_id"] == motor_id:
            return name


# Loading Motors Yaml File
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','config', 'head','motors.yaml'), 'r') as stream:
    try:
        head_dict = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        print(error)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','config', 'body','motors.yaml'), 'r') as stream:
    try:
        body_dict = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        print(error)

motors_dict = {}
motors_dict.update(head_dict['motors'])
motors_dict.update(body_dict['motors'])
