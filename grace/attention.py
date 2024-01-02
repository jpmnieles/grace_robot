import os
import sys
sys.path.append(os.getcwd())

import cv2
import dlib

from grace.utils import *


class PeopleAttention(object):

    def __init__(self) -> None:
        self.camera_mtx = load_camera_mtx()
        self.person_detected = False
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.path.join(os.getcwd(),'pretrained','shape_predictor_68_face_landmarks.dat'))
        dlib.cuda.set_device(0)

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
        x_target = landmarks.part(30).x
        y_target = landmarks.part(30).y
        delta_x = x_target - self.camera_mtx[eye]['cx']
        delta_y =  self.camera_mtx[eye]['cy'] - y_target
        return delta_x, delta_y
    
    def visualize_target(self, delta_x, delta_y, img, id:int, eye:str):
        """eye (str): select from ['left_eye', 'right_eye']
        """
        if eye == 'left_eye':
            detection = self.l_detections[id]
        elif eye == 'right_eye':
            detection = self.r_detections[id]
        cv2.rectangle(img, (detection.left(), detection.top()), (detection.right(), detection.bottom()), (0, 0, 255), 2)
        abs_x = self.camera_mtx[eye]['cx'] + delta_x
        abs_y = self.camera_mtx[eye]['cy'] - delta_y
        disp_img = cv2.drawMarker(img, (round(abs_x),round(abs_y)), color=(255, 0, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=13, thickness=2)
        return disp_img


class ChessboardAttention(object):

    def __init__(self) -> None:
        self.get_camera_mtx()

    def get_camera_mtx(self, filename='config/camera/camera_mtx.json'):
        with open(filename, 'r') as stream:
            data = json.load(stream)
            self.l_mtx = np.array(data['left_eye']['camera_matrix'])
            self.l_dist = np.array(data['left_eye']['distortion_coefficients'])
            self.r_mtx = np.array(data['right_eye']['camera_matrix'])
            self.r_dist = np.array(data['right_eye']['distortion_coefficients'])
            self.c_mtx = np.array(data['chest_cam']['camera_matrix'])
            self.c_dist = np.array(data['chest_cam']['distortion_coefficients'])
            return self.l_mtx, self.l_dist, self.r_mtx, self.r_dist, self.c_mtx, self.c_dist

    def visualize_chess_idx(self, chess_idx, chess_list, img):
        x, y = chess_list[chess_idx]
        img = cv2.drawMarker(img, (round(x),round(y)), color=(255, 0, 0), 
                       markerType=cv2.MARKER_TILTED_CROSS, markerSize=13, thickness=2)
        return img

    def process_img(self, chess_idx, img, camera):
        px = (None,None)
        chess_idx = 21

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            s_corners = corners2.squeeze()
            px = s_corners[chess_idx]
            img = self.visualize_chess_idx(chess_idx, s_corners, img)

        return px, img
