import os
import sys
sys.path.append(os.getcwd())

import cv2
import dlib
import numpy as np
import cv2.aruco as aruco

from grace.utils import *


class PeopleAttention(object):

    def __init__(self) -> None:
        self.camera_mtx = load_camera_mtx()
        self.person_detected = False
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.path.join(os.getcwd(),'pretrained','shape_predictor_68_face_landmarks.dat'))
        dlib.cuda.set_device(0)
        self.calib_params = load_json('config/calib/calib_params.json')

    def register_imgs(self, left_img, right_img):
        self.left_img = left_img
        self.right_img  = right_img

    def detect_people(self, left_img, right_img):
        # Detection       
        self.l_gray = cv2.cvtColor(left_img.copy(), cv2.COLOR_BGR2GRAY)
        self.r_gray = cv2.cvtColor(right_img.copy(), cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        self.l_detections = self.detector(self.l_gray, 0)
        self.r_detections = self.detector(self.r_gray, 0)

        # Person detected or not
        if len(self.l_detections) > 0 or len(self.r_detections) > 0:
            self.person_detected = True
            # print(self.camera_mtx)
        else:
            self.person_detected = False

    def get_pixel_target(self, id:int, eye:str):
        """id (int): select from [0, 1, ...]
        eye (str): select from ['left_eye', 'right_eye']
        """
        if eye == 'left_eye':
            detection = self.l_detections[id]
            img = self.l_gray
        elif eye == 'right_eye':
            detection = self.r_detections[id]
            img = self.r_gray
        landmarks = self.predictor(img, detection)
        x_target = landmarks.part(28).x
        y_target = landmarks.part(28).y
        delta_x = x_target - self.calib_params[eye]['x_center']
        delta_y =  self.calib_params[eye]['y_center'] - y_target
        return delta_x, delta_y
    
    def visualize_target(self, delta_x, delta_y, img, id:int, eye:str):
        """eye (str): select from ['left_eye', 'right_eye']
        """
        if eye == 'left_eye':
            detection = self.l_detections[id]
        elif eye == 'right_eye':
            detection = self.r_detections[id]
        cv2.rectangle(img, (detection.left(), detection.top()), (detection.right(), detection.bottom()), (0, 0, 255), 2)
        abs_x = self.calib_params[eye]['x_center'] + delta_x
        abs_y = self.calib_params[eye]['y_center'] - delta_y
        disp_img = cv2.drawMarker(img, (round(abs_x),round(abs_y)), color=(255, 0, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=13, thickness=2)
        return disp_img


class ChessboardAttention(object):

    def __init__(self) -> None:
        self.camera_mtx = load_camera_mtx()

    def visualize_chess_idx(self, px, img):
        x, y = px
        img = cv2.drawMarker(img, (round(x),round(y)), color=(255, 0, 0), 
                       markerType=cv2.MARKER_TILTED_CROSS, markerSize=13, thickness=2)
        return img

    def process_img(self, chess_idx, img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            s_corners = corners2.squeeze()
        else:
            s_corners = None

        return s_corners
    

class ChArucoAttention(object):

    def __init__(self) -> None:
        self.chess_squares = (9, 6)
        self.square_length = 0.06  # 60 mm
        self.marker_length = 0.051  # 51 mm
        
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.parameters = aruco.DetectorParameters()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.board = aruco.CharucoBoard(self.chess_squares, self.square_length, self.marker_length, self.dictionary)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def visualize_chess_idx(self, px, img):
        x, y = px
        img = cv2.drawMarker(img, (round(x), round(y)), color=(255, 0, 0), 
                             markerType=cv2.MARKER_TILTED_CROSS, markerSize=13, thickness=2)
        return img

    def process_img(self, chess_idx, img):
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization to enhance contrast
        equalized = cv2.equalizeHist(gray)

        binary = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY) 

        # Detect markers
        corners, ids, rejected = aruco.detectMarkers(binary, self.dictionary, parameters=self.parameters)
        
        if len(corners) > 0:
            charuco_retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, self.board)
        else:
            charuco_corners, charuco_ids = None, None

        return charuco_corners, charuco_ids
