import os
import sys
sys.path.append(os.getcwd())

import math
import time
import yaml
from datetime import datetime

import rospy
import message_filters
from hr_msgs.msg import TargetPosture, MotorStateList
from rospy_message_converter import message_converter
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, String, Header
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped

import tf
from tf.transformations import translation_matrix, quaternion_matrix
from sensor_msgs.msg import JointState

import cv2
import dlib
import numpy as np
import random
import threading
import copy

from grace.utils import *
from aec.baseline import BaselineCalibration
from grace.attention import *


CALIB_PARAMS = load_json('config/calib/calib_params.json')
CAMERA_MTX = load_json("config/camera/camera_mtx.json")
LEFT_EYE_CAMERA_MTX = np.array(CAMERA_MTX['left_eye']['camera_matrix'])
LEFT_EYE_DIST_COEF = np.array(CAMERA_MTX['left_eye']['distortion_coefficients']).squeeze()
RIGHT_EYE_CAMERA_MTX = np.array(CAMERA_MTX['right_eye']['camera_matrix'])
RIGHT_EYE_DIST_COEF = np.array(CAMERA_MTX['right_eye']['distortion_coefficients']).squeeze()
CHEST_CAM_CAMERA_MTX = np.array(CAMERA_MTX['chest_cam']['camera_matrix'])
CHEST_CAM_DIST_COEF = np.array(CAMERA_MTX['chest_cam']['distortion_coefficients']).squeeze()


attention = ExpChArucoAttention()

def get_obj_pts_chest(img, camera_mtx, dist_coef):
    # Input
    image_copy = np.copy(img)
    
    # Function
    charuco_corners, charuco_ids, marker_corners, marker_ids = attention.charuco_detector.detectBoard(image_copy)
    if not (charuco_ids is None) and len(charuco_ids) > 0:
        if len(camera_mtx) > 0 and len(charuco_ids) >= 4:
            try:
                obj_points, img_points = attention.board.matchImagePoints(charuco_corners, charuco_ids)
                flag, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_mtx, dist_coef)
                if flag:
                    # Convert the rotation vector to a rotation matrix
                    rotation_matrix, _ = cv2.Rodrigues(rvec)

                    # Homogeneous Coordinates (H = Tbc =T_board2cam)
                    T = np.eye(4)
                    T[:3,:3] = rotation_matrix
                    T[:3,-1] = tvec.T

            except cv2.error as error_inst:
                print("SolvePnP recognize calibration pattern as non-planar pattern. To process this need to use "
                      "minimum 6 points. The planar pattern may be mistaken for non-planar if the pattern is "
                      "deformed or incorrect camera parameters are used.")
                print(error_inst.err)
                T = None
                
    # Mapping
    charuco_id2pts = dict(zip(charuco_ids.flatten(),obj_points.squeeze()))
    
    # Defined Targets
    charuco_targets = [62,63,
                       76,77]
    
    # Get Projections
    T_bc = T
    proj_pts_list = []
    for x in charuco_targets:
        proj_pts = T_bc @ np.array([charuco_id2pts[x][0],charuco_id2pts[x][1],0,1])
        proj_pts_list.append(proj_pts)
    
    return proj_pts_list


class VisuoMotorNode(object):

    rl_state = {  # RL environment on the network takes care of the other side
        'chess_idx': None,
        'theta_left_pan': None,
        'theta_right_pan': None,
        'theta_tilt': None,
        'chest_cam_px_x_tminus1': None,
        'chest_cam_px_y_tminus1': None,  
        'left_eye_px_x_tminus1': None,
        'left_eye_px_y_tminus1': None,
        'right_eye_px_x_tminus1': None,
        'right_eye_px_y_tminus1': None,
        'chest_cam_px_x': None,
        'chest_cam_px_y': None,  
        'left_eye_px_x': None,
        'left_eye_px_y': None,
        'right_eye_px_x': None,
        'right_eye_px_y': None,
        'dx_l': None,
        'dy_l': None,
        'dx_r': None,
        'dy_r': None,
        '3d_point': None,
        'chest_pan_angle': None,
        'chest_tilt_angle': None,
        'plan_phi_left_pan': None,
        'plan_phi_right_pan': None,
        'plan_phi_tilt': None,
        'chest_img': None,
        'left_eye_img': None,
        'right_eye_img': None,
        'depth_img': None,
        'chest_img_stamp': None,
        'left_eye_img_stamp': None,
        'right_eye_img_stamp': None,
        'depth_img_stamp': None,
        'theta_left_pan_cmd': None,
        'theta_right_pan_cmd': None,
        'theta_tilt_cmd': None,
    }

    joints_list = ['neck_roll', 'neck_pitch', 'neck_yaw',
                   'head_roll', 'head_pitch', 
                   'eyes_pitch', 'lefteye_yaw', 'righteye_yaw']


    def __init__(self, motors=["EyeTurnLeft", "EyeTurnRight", "EyesUpDown",
                               "NeckRotation", "UpperGimbalLeft", "UpperGimbalRight",
                               "LowerGimbalLeft", "LowerGimbalRight"], degrees=True):
        self.motor_lock = threading.Lock()
        self.buffer_lock = threading.Lock()
        self.action_lock = threading.Lock()
        self.motors = motors
        self.degrees = degrees
        self._set_motor_limits(motors)

        self.attention = PeopleAttention()
        self.calibration = BaselineCalibration(self.buffer_lock)  # RL Model
        self.calibration.toggle_backlash(True)
        self.frame_stamp_tminus1 = rospy.Time.now()
        self.motor_stamp_tminus1 = rospy.Time.now()
        self.chess_idx_tminus1 = 0

        self.motor_pub = rospy.Publisher('/hr/actuators/pose', TargetPosture, queue_size=1)
        self.camera_mtx = load_json("config/camera/camera_mtx.json")
        self.camera_mtx_params = load_camera_mtx()
        self.bridge = CvBridge()
        time.sleep(1)
        
        # Initial Reset Action
        self.move((0, 0, 0, 0, 44, -44, -13, 13))
        time.sleep(3.0)
        self.move((0, 0, 0, 0, 0, 0, 0, 0))
        time.sleep(3.0)
        self.move((0, 0, 0, -35, 0, 0, 0, 0))
        time.sleep(3.0)
        self.move((0, 0, 0, 0, 0, 0, 0, 0))
        time.sleep(3.0)
        self.move((-18, -18, 22))
        time.sleep(1.0)
        self.move((0, 0, 0))
        time.sleep(1.0)


        self.action = None

        self.motor_sub = rospy.Subscriber('/hr/actuators/motor_states', MotorStateList, self._capture_state)
        self.left_eye_sub = message_filters.Subscriber("/left_eye/image_raw", Image)
        self.right_eye_sub = message_filters.Subscriber("/right_eye/image_raw", Image)  # TODO: change to right eye when there is better camera
        self.chest_cam_sub = message_filters.Subscriber('/hr/perception/jetson/realsense/camera/color/image_raw', Image)
        self.depth_cam_sub = message_filters.Subscriber('/hr/perception/jetson/realsense/camera/aligned_depth_to_color/image_raw', Image)
        self.ats = message_filters.ApproximateTimeSynchronizer([self.left_eye_sub, self.right_eye_sub, 
                                                                self.chest_cam_sub, self.depth_cam_sub], queue_size=1, slop=0.25)
        self.ats.registerCallback(self.eye_imgs_callback)
        self.rt_display_pub = rospy.Publisher('/output_display1', Image, queue_size=1)
        # self.point_pub = rospy.Publisher('/point_location', PointStamped, queue_size=1)
        # self.tf_listener = tf.TransformListener()

        self.key_press_sub = rospy.Subscriber('key_press', String, self._key_press_callback)

        self.chess_idx = 0
        self.ctr = 0
        self.disp_img = np.zeros((480,640,3), dtype=np.uint8)
        self.calib_params = load_json('config/calib/calib_params.json')

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.path.join(os.getcwd(),'pretrained','shape_predictor_68_face_landmarks.dat'))
        dlib.cuda.set_device(0)

        rospy.loginfo('Running')

    def set_action(self, action):
        with self.action_lock:
            self.action = action

    def _key_press_callback(self, event):
        if event.data == '`':  # Replace '`' with the desired key  # Lowercase tilde (~)
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            results_dir = os.path.join(parent_dir, 'results','capture_eyes_chest_cam')
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
                print(f"Directory created: {results_dir}")
            else:
                print(f"Directory already exists: {results_dir}")
            
            dt_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            title_str = 'captureeyeschestcam'
            json_path = os.path.join(results_dir, dt_str+'_'+title_str+'_json'+'.json')
            chestimg_path = os.path.join(results_dir, dt_str+'_'+title_str+'_chestimg'+'.png')
            leftimg_path = os.path.join(results_dir, dt_str+'_'+title_str+'_leftimg'+'.png')
            rightimg_path = os.path.join(results_dir, dt_str+'_'+title_str+'_rightimg'+'.png')
            
            # Saving images
            cv2.imwrite(chestimg_path, self.chest_img)
            cv2.imwrite(leftimg_path, self.left_img)
            cv2.imwrite(rightimg_path, self.right_img)

    def _capture_state(self, msg):
        """Callback for capturing motor state
        """
        updated_motors_list = []
        self._msg = msg
        for idx, motor_msg in enumerate(msg.motor_states):
            if motor_msg.name in self.names:
                idx = self.motors.index(motor_msg.name)
                self._motor_states[idx] = message_converter.convert_ros_message_to_dictionary(motor_msg)
                self._motor_states[idx]['angle'] = motor_int_to_angle(motor_msg.name, motor_msg.position, self.degrees)
                updated_motors_list.append(motor_msg.name)
        # Debug
        # print('===>:', updated_motors_list)

        
    def depth_to_pointcloud(self, px, depth_img, camera_mtx, z_replace=1.0):  
        fx = camera_mtx[0][0]
        cx = camera_mtx[0][2]
        fy = camera_mtx[1][1]
        cy = camera_mtx[1][2]
        u = round(px[0])
        v = round(px[1])
        z = depth_img[v,u]/1000.0
        z = z - 0.0042  # Added realsense discrepancy
        if z<=0:
            z = z_replace
        x = ((u-cx)/fx)*z
        y = ((v-cy)/fy)*z
        return x,y,z

    def transform_points(self, pts, T_mtx):
        new_obj_pts = []
        for pt in pts:
            temp_pt = np.append(pt, 1).reshape(-1,1)
            temp_pt2 = (T_mtx @ temp_pt).squeeze()
            new_obj_pts.append(temp_pt2[:3])
        new_obj_pts = np.array(new_obj_pts)
        return new_obj_pts


    def eye_imgs_callback(self, left_img_msg, right_img_msg, chest_img_msg, depth_img_msg):
        # Motor Trigger Sync (3.33 FPS or 299.99 ms)
        max_stamp = max(left_img_msg.header.stamp, right_img_msg.header.stamp, chest_img_msg.header.stamp)
        elapsed_time = (max_stamp - self.frame_stamp_tminus1).to_sec()
        if elapsed_time > 285e-3:
            start = time.time()
            print('--------------')
            rospy.loginfo(f'FPS: {1/elapsed_time: .{2}f}')
            self.frame_stamp_tminus1 = max_stamp

            # Conversion of ROS Message
            self.left_img = self.bridge.imgmsg_to_cv2(left_img_msg, "bgr8")
            self.right_img = self.bridge.imgmsg_to_cv2(right_img_msg, "bgr8")
            self.chest_img = self.bridge.imgmsg_to_cv2(chest_img_msg, "bgr8")
            self.depth_img = self.bridge.imgmsg_to_cv2(depth_img_msg, "16UC1")
            depth_map_normalized = copy.deepcopy(self.depth_img) / self.depth_img.max()  # Normalize values to [0, 1]
            gray_depth_img = (depth_map_normalized * 255).astype(np.uint8)
            gray_depth_img = cv2.cvtColor(gray_depth_img, cv2.COLOR_GRAY2BGR)
            # print(left_img_msg.header, right_img_msg.header, chest_img_msg.header)
            # print(self.depth_img)
            # print('Median depth (m):', np.median(self.depth_img)/1000.0)
            print('Median depth of ROI(m):', np.median(self.depth_img[69:221,190:664])/1000.0)

            ## Attention ##
            chest_disp = copy.deepcopy(self.chest_img)
            charuco_corners, charuco_ids, marker_corners, marker_ids = attention.charuco_detector.detectBoard(chest_disp)
            if not (charuco_ids is None) and len(charuco_ids) > 0:
                if len(CHEST_CAM_CAMERA_MTX) > 0 and len(charuco_ids) >= 4:
                    try:
                        obj_points, img_points = attention.board.matchImagePoints(charuco_corners, charuco_ids)
                        flag, rvec, tvec = cv2.solvePnP(obj_points, img_points, CHEST_CAM_CAMERA_MTX, CHEST_CAM_DIST_COEF)
                        if flag:
                            # Convert the rotation vector to a rotation matrix
                            rotation_matrix, _ = cv2.Rodrigues(rvec)

                            # Homogeneous Coordinates (H = Tbc =T_board2cam)
                            T = np.eye(4)
                            T[:3,:3] = rotation_matrix
                            T[:3,-1] = tvec.T

                    except cv2.error as error_inst:
                        print("SolvePnP recognize calibration pattern as non-planar pattern. To process this need to use "
                            "minimum 6 points. The planar pattern may be mistaken for non-planar if the pattern is "
                            "deformed or incorrect camera parameters are used.")
                        print(error_inst.err)
                        T = None
                        
            # Mapping
            charuco_id2pts = dict(zip(charuco_ids.flatten(),obj_points.squeeze()))
            
            # Defined Targets
            charuco_targets = [48,49,50,
                                62,63,64,
                                76,77,78]
            charuco_target_idx = [charuco_ids.flatten().tolist().index(x) for x in charuco_targets]
            
            # Charuco Display
            if not (charuco_ids is None) and len(charuco_ids) > 0:
                chest_disp = aruco.drawDetectedCornersCharuco(chest_disp, np.array([charuco_corners[i] for i in charuco_target_idx]).reshape(-1,1,2),
                                                               np.array(charuco_targets).reshape(-1,1))
            
            # Get Projections
            T_bc = T
            proj_pts_list = []
            for x in charuco_targets:
                proj_pts = T_bc @ np.array([charuco_id2pts[x][0],charuco_id2pts[x][1],0,1])
                proj_pts_list.append(proj_pts)

            print(f"Charuco 48 depth (m): {proj_pts_list[0][2]}")
            print(f"Charuco 50 depth (m): {proj_pts_list[2][2]}")
            print(f"Charuco 76 depth (m): {proj_pts_list[6][2]}")
            print(f"Charuco 78 depth (m): {proj_pts_list[8][2]}")

            # Visualization
            left_img = self.ctr_cross_img(copy.deepcopy(self.left_img), 'left_eye')
            right_img = self.ctr_cross_img(copy.deepcopy(self.right_img), 'right_eye')
            chest_img = self.ctr_cross_img(copy.deepcopy(self.chest_img), 'chest_cam')
            # concat_img = np.hstack((chest_img, left_img, right_img))
            chest_roi_img = cv2.rectangle(copy.deepcopy(chest_disp), (345,122), (545,322), (0, 255, 0), 2)
            concat_img = np.hstack((chest_roi_img, left_img, right_img))

            # Resizing
            height, width = concat_img.shape[:2]
            concat_img = cv2.resize(concat_img, (round(width/2), round(height/2)))

            # Output Display 1
            self.rt_display_pub.publish(self.bridge.cv2_to_imgmsg(concat_img, encoding="bgr8"))

            # Reassigning
            self.chess_idx_tminus1 = self.chess_idx

    def _capture_limits(self, motor):
        int_min = motors_dict[motor]['motor_min']
        int_init = motors_dict[motor]['init']
        int_max = motors_dict[motor]['motor_max']
        angle_min = motor_int_to_angle(motor, int_min, self.degrees)
        angle_init = motor_int_to_angle(motor, int_init, self.degrees)
        angle_max = motor_int_to_angle(motor, int_max, self.degrees)
        limits = {'int_min': int_min, 
                  'int_init': int_init, 
                  'int_max': int_max,
                  'angle_min': angle_min, 
                  'angle_init': angle_init, 
                  'angle_max': angle_max}
        return limits

    def _check_limits(self, name, value):
        if value < self._motor_limits[name]['angle_min']:
            value = self._motor_limits[name]['angle_min']
        elif value > self._motor_limits[name]['angle_max']:
            value = self._motor_limits[name]['angle_max']
        return value

    def _set_motor_limits(self, names: list):
        self.names = names
        self.num_names = len(self.names)
        self._motor_limits = {motor: self._capture_limits(motor) for motor in self.names}
        self._motor_states = [None]*self.num_names
    
    def move(self, values):
        values = [self._check_limits(self.names[i],x) for i,x in enumerate(values)]
        if self.degrees:
            values = [math.radians(x) for x in values]
        args = {"names":self.names, "values":values}
        self.motor_pub.publish(TargetPosture(**args))

    def move_specific(self, names, values):
        values = [self._check_limits(names[i],x) for i,x in enumerate(values)]
        if self.degrees:
            values = [math.radians(x) for x in values]
        args = {"names":names, "values":values}
        self.motor_pub.publish(TargetPosture(**args))

    def ctr_cross_img(self, img, eye:str):
        """eye (str): select from ['left_eye', 'right_eye', 'chest_cam']
        """
        if eye == 'chest_cam':
            # True Optical Center
            img = cv2.putText(img, eye, (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                              1, (0,255,0), 2, cv2.LINE_AA)
            img = cv2.line(img, (round(self.camera_mtx_params[eye]['cx']), 0), (round(self.camera_mtx_params[eye]['cx']), 480), (0,255,0))
            img = cv2.line(img, (0, round(self.camera_mtx_params[eye]['cy'])), (848, round(self.camera_mtx_params[eye]['cy'])), (0,255,0))
            img = cv2.drawMarker(img, (round(self.camera_mtx_params[eye]['cx']), round(self.camera_mtx_params[eye]['cy'])), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        else:
            img = cv2.putText(img, eye, (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                              1, (0,255,0), 2, cv2.LINE_AA)
            img = cv2.line(img, (round(self.calib_params[eye]['x_center']), 0), (round(self.calib_params[eye]['x_center']), 480), (0,255,0))
            img = cv2.line(img, (0, round(self.calib_params[eye]['y_center'])), (640, round(self.calib_params[eye]['y_center'])), (0,255,0))
            img = cv2.drawMarker(img, (round(self.calib_params[eye]['x_center']), round(self.calib_params[eye]['y_center'])), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        return img


if __name__ == '__main__':
    rospy.init_node('visuomotor')
    vismotor = VisuoMotorNode()
    rospy.spin()
