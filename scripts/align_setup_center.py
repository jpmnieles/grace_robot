import os
import sys
sys.path.append(os.getcwd())

import time
import json
import datetime
import numpy as np
import cv2 as cv

from grace.camera import LeftEyeCapture, RightEyeCapture
from grace.control import ROSMotorClient


class GraceAlign(object):


    SQUARE_LENGTH = 47
    NO_ROTATION = np.eye(3)


    def __init__(self, epsilon=0.03) -> None:
        self.epsilon = epsilon
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((6*9,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) * self.SQUARE_LENGTH
        self.axis = np.float32([[2,0,0], [0,2,0], [0,0,-2]]).reshape(-1,3) * self.SQUARE_LENGTH  # Vector direction for display

    
    def get_camera_mtx(self, filename='config/camera/camera_mtx.json'):
        with open(filename, 'r') as stream:
            data = json.load(stream)
            l_mtx = np.array(data['left_eye']['camera_matrix'])
            l_dist = np.array(data['left_eye']['distortion_coefficients'])
            r_mtx = np.array(data['right_eye']['camera_matrix'])
            r_dist = np.array(data['right_eye']['distortion_coefficients'])
            return l_mtx, l_dist, r_mtx, r_dist    


    def rmse(self, actual, predicted):
        diff=np.subtract(actual,predicted)
        square=np.square(diff)
        mse=square.mean()
        rmse=np.sqrt(mse)
        return rmse
    

    def draw_axis(self, img, corners, imgpts):
        corner = tuple(corners[0].astype(int).ravel())
        img = cv.line(img, corner, tuple(imgpts[0].astype(int).ravel()), (255,0,0), 5)
        img = cv.line(img, corner, tuple(imgpts[1].astype(int).ravel()), (0,255,0), 5)
        img = cv.line(img, corner, tuple(imgpts[2].astype(int).ravel()), (0,0,255), 5)
        return img


    def process_img(self, img, mtx, dist, camera):
        chess_idx = 21

        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (9,6),None)

        # Putting the Center
        if camera=='left':
            img = cv.line(img, (315, 0), (315, 480), (0,255,0))
            img = cv.line(img, (0, 202), (640, 202), (0,255,0))
        elif camera=='right':
            img = cv.line(img, (302, 0), (302, 480), (0,255,0))
            img = cv.line(img, (0, 219), (640, 219), (0,255,0))

        if ret == True:
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), self.criteria)
            s_corners = corners2.squeeze()
            
            # Find the rotation and translation vectors.
            ret, rvecs, tvecs = cv.solvePnP(self.objp, corners2, mtx, dist)
            rmat,_ = cv.Rodrigues(rvecs)
            imgpts, jac = cv.projectPoints(self.axis, rvecs, tvecs, mtx, dist)
            img = self.draw_axis(img,corners2,imgpts)
            
            # Putting Text Labels
            img = cv.putText(img=img, text="Rotation Matrix:", org=(5, 425), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color = (0, 0, 255), thickness=2)
            img = cv.putText(img=img, text=str(rmat[0]), org=(5, 440), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color = (0, 0, 255), thickness=2)
            img = cv.putText(img=img, text=str(rmat[1]), org=(5, 455), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color = (0, 0, 255), thickness=2)
            img = cv.putText(img=img, text=str(rmat[2]), org=(5, 470), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color = (0, 0, 255), thickness=2)

            # 9x6 Chessboard Center Position
            img = cv.putText(img=img, text=f"{s_corners[chess_idx]}", org=(200, 50), fontFace=cv.FONT_HERSHEY_SIMPLEX,  # 21 to 12
                                fontScale=0.6, color = (0, 255, 0), thickness=2)

            # Checking for Alignment
            error = self.rmse(self.NO_ROTATION, rmat)
            img = cv.putText(img=img, text=f"RMSE: %.4f" % (error), org=(520, 470), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color = (0, 0, 255), thickness=2)
            if error < self.epsilon:
                img = cv.putText(img=img, text="ALIGNED", org=(260, 30), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                fontScale=1, color = (0, 255, 0), thickness=2)
        return img


if __name__ == "__main__":
    # Initialization
    left_cam = LeftEyeCapture()
    right_cam = RightEyeCapture()
    alignment = GraceAlign(epsilon=0.05)
    motor_client = ROSMotorClient(["EyeTurnLeft", "EyeTurnRight", "EyesUpDown"], degrees=True, debug=False)

    # Load Camera Matrix
    l_mtx, l_dist, r_mtx, r_dist = alignment.get_camera_mtx("config/camera/camera_mtx.json")

    # Reset
    motor_client.move([0,0,0])
    time.sleep(0.3333)
    motor_client.slow_move(idx=0,position=-8,step_size=0.0879,time_interval=0.015)
    motor_client.slow_move(idx=0,position=0,step_size=0.0879,time_interval=0.015)
    motor_client.slow_move(idx=1,position=-8,step_size=0.0879,time_interval=0.015)
    motor_client.slow_move(idx=1,position=0,step_size=0.0879,time_interval=0.015)

    # Loop
    while(1):
        try:
            l_img = left_cam.frame
            l_img = alignment.process_img(l_img, l_mtx, l_dist, camera="left")
            cv.imshow("Left Camera", l_img)
        except:
            pass
        
        try:
            r_img = right_cam.frame
            r_img = alignment.process_img(r_img, r_mtx, r_dist, camera="right")
            cv.imshow("Right Camera", r_img)
        except:
            pass       

        # Key Press  
        key = cv.waitKey(1)
        if key == 108:  # letter l
            """Save left eye camera image. Press 'l' key
            """
            frame = left_cam.frame
            print("Left Eye Camera Shape:", frame.shape)
            date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") 
            fn_str = 'results/' + date_str + '_left_eye.png'
            cv.imwrite(fn_str, frame)
            print("Saving Left Camera Image to: ", fn_str)
        elif key == 114:  # letter r
            """Save right eye camera image. Press 'r' key
            """
            frame = right_cam.frame
            print("Right Eye Camera:", frame.shape)
            date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") 
            fn_str = 'results/' + date_str + '_right_eye.png'
            cv.imwrite(fn_str, frame)
            print("Saving Left Camera Image to: ", fn_str)
        elif key == 115:  # letter s
            """Save both left & right eye camera image. Press 'r' key
            """
            l_frame = left_cam.frame
            r_frame = right_cam.frame
            print("Left Eye Camera Shape:", l_frame.shape)
            print("Right Eye Camera:", r_frame.shape)
            date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") 
            l_fn_str = 'results/' + date_str + '_left_eye.png'
            r_fn_str = 'results/' + date_str + '_right_eye.png'
            cv.imwrite(l_fn_str, l_frame)
            cv.imwrite(r_fn_str, r_frame)
            print("Saving Left Camera Image to: ", l_fn_str)
            print("Saving Right Camera Image to: ", r_fn_str)
        elif key == 99:  # letter c
            """Resets the eye position via slow movements. Press 'c' key
            """
            motor_client.move([0,0,0])
            time.sleep(0.3333)
            motor_client.slow_move(idx=0,position=-18,step_size=0.0879,time_interval=0.015)
            motor_client.slow_move(idx=0,position=0,step_size=0.0879,time_interval=0.015)
            motor_client.slow_move(idx=1,position=-18,step_size=0.0879,time_interval=0.015)
            motor_client.slow_move(idx=1,position=0,step_size=0.0879,time_interval=0.015)
        elif key == 27:  # Esc
            """Execute end of program. Press 'esc' key to escape program
            """
            del left_cam
            del right_cam
            break