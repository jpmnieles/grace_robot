import os
import sys
sys.path.append(os.getcwd())

import math
import time
import yaml
import pandas as pd
import pickle
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

def get_charuco_camera_pose(img, camera_mtx, dist_coef):
    charuco_corners, charuco_ids, marker_corners, marker_ids = attention.charuco_detector.detectBoard(img)
    if not (charuco_ids is None) and len(charuco_ids) >= 4:
        try:
            obj_points, img_points = attention.board.matchImagePoints(charuco_corners, charuco_ids)
            flag, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_mtx, dist_coef)

            # Convert the rotation vector to a rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)
    
            # Homogeneous Coordinates
            H = np.eye(4)
            H[:3,:3] = rotation_matrix
            H[:3,-1] = tvec.T
    
            # Camera Pose
            T = np.linalg.inv(H)
        
        except cv2.error as error_inst:
            print("SolvePnP recognize calibration pattern as non-planar pattern. To process this need to use "
                  "minimum 6 points. The planar pattern may be mistaken for non-planar if the pattern is "
                  "deformed or incorrect camera parameters are used.")
            print(error_inst.err)
            T = None
    else:
        T = None

    return T

def get_world_deproject(eye, camera_mtx, dist_coef, T_be):
    # Foveal Center
    gx = CALIB_PARAMS[eye]['x_center']
    gy = CALIB_PARAMS[eye]['y_center']

    # Get Normalized Camera Coordinates
    x_prime, y_prime = cv2.undistortPoints((gx,gy),   # x' = X_C/Z_C
                        camera_mtx,                   # y' = Y_C/Z_C
                        dist_coef).squeeze()

    # Inverse of T_be
    T = np.linalg.inv(T_be)

    # System of Linear Equations
    a = np.array([[T[0,0],T[0,1],-x_prime], 
                  [T[1,0],T[1,1],-y_prime],
                  [T[2,0],T[2,1],-1.0]])
    b = np.array([-T[0,3],-T[1,3],-T[2,3]])
    
    # Deprojection
    x = np.linalg.solve(a,b)  # X_W, Y_W, Z_C
    world_pts = np.array([x[0],x[1],0])  # X_W, Y_W, Z_W

    return world_pts

def transform_points(pts, T_mtx):
    new_obj_pts = []
    for pt in pts:
        temp_pt = np.append(pt, 1).reshape(-1,1)
        temp_pt2 = (T_mtx @ temp_pt).squeeze()
        new_obj_pts.append(temp_pt2[:3])
    new_obj_pts = np.array(new_obj_pts)
    return new_obj_pts

def chest_rgb_to_eye(T_chest, T_input):
    T_chest_inv = np.linalg.inv(T_chest)  # T_cb = T_bc^(-1)
    T_ce = np.matmul(T_chest_inv, T_input)  # T_ce = T_cb x T_be
    rvec, _ = cv2.Rodrigues(T_ce[:3,:3])
    tvec = T_ce[:3,3]
    return T_ce, rvec.flatten(), tvec.flatten()


class HeadEyesSweepNode(object):

    rl_state = {  # RL environment on the network takes care of the other side
        'theta_left_pan': None,
        'theta_right_pan': None,
        'theta_tilt': None,
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

    motor_state = {
        "EyeTurnLeft": None,
        "EyeTurnRight": None, 
        "EyesUpDown": None,
        "NeckRotation": None,
        "UpperGimbalLeft": None,
        "UpperGimbalRight": None,
        "LowerGimbalLeft": None,
        "LowerGimbalRight": None
    }
    
    pickle_data = {
        'subject_num': None,
        'markers': None,
        'trial_num': None,
        'data': [],
    }

    def __init__(self, list_ep=list(range(-14,15,2)), 
                 list_et=list(range(20,-31,-5)),
                 list_lnp=list(range(-35,36,5)), 
                 list_lnt=list(range(-10,31,10)),
                 list_unt=list(range(40,-11,-10)),
                 motors=["EyeTurnLeft", "EyeTurnRight", "EyesUpDown",
                         "NeckRotation", "UpperGimbalLeft", "UpperGimbalRight",
                         "LowerGimbalLeft", "LowerGimbalRight"],
                 degrees=True):
        # Timer Start
        self.start = time.time()
        rospy.loginfo('Starting')

        # Locks
        self.motor_lock = threading.Lock()
        self.buffer_lock = threading.Lock()
        self.left_cam_lock = threading.Lock()
        self.right_cam_lock = threading.Lock()
        self.chest_cam_lock = threading.Lock()
        self.depth_cam_lock = threading.Lock()

        # Inputs
        self.list_ep = list_ep
        self.list_et = list_et
        self.list_lnp = list_lnp
        self.list_lnt = list_lnt
        self.list_unt = list_unt

        # Initialization
        self.motors = motors
        self.degrees = degrees
        self._set_motor_limits(motors)
        self.camera_mtx = load_camera_mtx()
        self.calib_params = load_json('config/calib/calib_params.json')
        self.motor_pub = rospy.Publisher('/hr/actuators/pose', TargetPosture, queue_size=1)
        self.rt_display_pub = rospy.Publisher('/output_display1', Image, queue_size=1)
        self.bridge = CvBridge()
        time.sleep(1)
        
        # Initial Reset Action
        self.move((0, 0, 0, 0, 44, -44, -13, 13))
        time.sleep(3.0)
        self.move((0, 0, 0, 0, 0, 0, 0, 0))
        time.sleep(3.0)
        self.move((0, 0, 0, -40, 0, 0, 0, 0))
        time.sleep(3.0)
        self.move((0, 0, 0, 0, 0, 0, 0, 0))
        time.sleep(3.0)
        self.move((-18, -18, 22))
        time.sleep(1.0)
        self.move((0, 0, 0))
        time.sleep(1.0)

        # Subscribers
        self.motor_sub = rospy.Subscriber('/hr/actuators/motor_states', MotorStateList, self._capture_state)
        self.left_cam_sub = rospy.Subscriber('/left_eye/image_raw', Image, self.left_img_callback)
        self.right_cam_sub = rospy.Subscriber('/right_eye/image_raw', Image, self.right_img_callback)
        self.chest_cam_sub = rospy.Subscriber('/hr/perception/jetson/realsense/camera/color/image_raw', 
                                              Image, self.chest_img_callback)
        self.depth_cam_sub = rospy.Subscriber('/hr/perception/jetson/realsense/camera/aligned_depth_to_color/image_raw', 
                                              Image, self.depth_img_callback)
        time.sleep(3.0)

    def _capture_state(self, msg):
        """Callback for capturing motor state
        """
        self._msg = msg
        with self.motor_lock:
            for idx, motor_msg in enumerate(msg.motor_states):
                if motor_msg.name in self.names:
                    idx = self.motors.index(motor_msg.name)
                    self._motor_states[idx] = message_converter.convert_ros_message_to_dictionary(motor_msg)
                    self._motor_states[idx]['angle'] = motor_int_to_angle(motor_msg.name, motor_msg.position, self.degrees)
      
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

    def left_img_callback(self, left_img_msg):
        with self.left_cam_lock:
            self.left_img = self.bridge.imgmsg_to_cv2(left_img_msg, "bgr8")
    
    def right_img_callback(self, right_img_msg):
        with self.right_cam_lock:
            self.right_img = self.bridge.imgmsg_to_cv2(right_img_msg, "bgr8")

    def chest_img_callback(self, chest_img_msg):
        with self.chest_cam_lock:
            self.chest_img = self.bridge.imgmsg_to_cv2(chest_img_msg, "bgr8")

    def depth_img_callback(self, depth_img_msg):
        with self.depth_cam_lock:
            self.depth_raw = self.bridge.imgmsg_to_cv2(depth_img_msg, "16UC1")
            depth_norm = 255*(copy.deepcopy(self.depth_raw) / self.depth_raw.max())
            depth_norm = depth_norm.astype(np.uint8)
            self.depth_img = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    def run(self):
        rospy.loginfo('Running')

        # Create Directory
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(parent_dir, 'results','headeyes_pantilt_sweep_charuco')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            print(f"Directory created: {results_dir}")
        else:
            print(f"Directory already exists: {results_dir}")
        
        # Create Subdirectory
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        sub_dir = os.path.join(results_dir, dt_str)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
            print(f"Subdirectory created: {sub_dir}")
        else:
            print(f"Subdirectory already exists: {sub_dir}")

        # Total Counter
        ctr = 0
        total_ctr = len(self.list_ep)*len(self.list_et)*len(self.list_lnp)*len(self.list_lnt)*len(self.list_unt)

        # Get Images
        global left_img, left_img, right_img, chest_img
        global depth_raw, depth_img

        # Neck
        for lnt in self.list_lnt:
            for unt in self.list_unt:
                for i in range(2):
                    if i==0:
                        self.move_specific(["UpperGimbalLeft","UpperGimbalRight","LowerGimbalLeft","LowerGimbalRight"],
                                           [44,-44,-13,13])
                        rospy.loginfo('lnt & unt reset')
                        rospy.sleep(1.5)
                    else:
                        self.move_specific(["UpperGimbalLeft","UpperGimbalRight","LowerGimbalLeft","LowerGimbalRight"],
                                           [unt,-unt,lnt,-lnt])
                        rospy.loginfo('lnt:%d, unt:%d' % (lnt,unt))
                        rospy.sleep(3)
                    
                # Neck Rotation
                for lnp in self.list_lnp:
                    for j in range(2):
                        if j==0:
                            self.move_specific(["NeckRotation"],[-40])
                            rospy.loginfo('lnp reset')
                            rospy.sleep(1.5)
                        else:
                            self.move_specific(["NeckRotation"],[lnp])
                            rospy.loginfo('lnp:%d' % (lnp))
                            rospy.sleep(3)

                    # Initialization
                    left_headeyes_dict = {'x_c': [],
                                          'y_c': [],
                                          'z_c': [],
                                          'cmd_theta_lower_neck_pan':[],
                                          'cmd_theta_lower_neck_tilt':[],
                                          'cmd_theta_upper_neck_tilt':[],
                                          'cmd_theta_left_eye': [],
                                          'cmd_theta_tilt':[],
                                          'state_theta_lower_neck_pan':[],
                                          'state_theta_left_lower_neck_tilt':[],
                                          'state_theta_right_lower_neck_tilt':[],
                                          'state_theta_left_upper_neck_tilt':[],
                                          'state_theta_right_upper_neck_tilt':[],
                                          'state_theta_left_eye': [],
                                          'state_theta_tilt':[],
                                          'rvec_0': [],
                                          'rvec_1': [],
                                          'rvec_2': [],
                                          'tvec_0': [],
                                          'tvec_1': [],
                                          'tvec_2': [],
                                          }
                    
                    right_headeyes_dict = {'x_c': [],
                                           'y_c': [],
                                           'z_c': [],
                                           'cmd_theta_lower_neck_pan':[],
                                           'cmd_theta_lower_neck_tilt':[],
                                           'cmd_theta_upper_neck_tilt':[],
                                           'cmd_theta_right_eye': [],
                                           'cmd_theta_tilt':[],
                                           'state_theta_lower_neck_pan':[],
                                           'state_theta_left_lower_neck_tilt':[],
                                           'state_theta_right_lower_neck_tilt':[],
                                           'state_theta_left_upper_neck_tilt':[],
                                           'state_theta_right_upper_neck_tilt':[],
                                           'state_theta_right_eye': [],
                                           'state_theta_tilt':[],
                                           'rvec_0': [],
                                           'rvec_1': [],
                                           'rvec_2': [],
                                           'tvec_0': [],
                                           'tvec_1': [],
                                           'tvec_2': [],
                                          }

                    eye_ctr = 0
                    total_eye_ctr = len(self.list_ep)*len(self.list_et)
                    eye_start = time.time()

                    # Eyes
                    for et in self.list_et:
                        for ep in self.list_ep:             
                            for k in range(2):
                                if k==0:
                                    # Reset
                                    self.move_specific(["EyeTurnLeft","EyeTurnRight","EyesUpDown"],
                                                        [-18,-18,22])
                                    rospy.loginfo('et & ep reset')
                                    rospy.sleep(0.5)
                                else:
                                    self.move_specific(["EyeTurnLeft", "EyeTurnRight", "EyesUpDown"],
                                                        [ep,ep,et])
                                    rospy.loginfo('et:%d, ep:%d' % (et,ep))
                                    rospy.sleep(1)
                                
                            # Capture
                            with self.left_cam_lock:
                                left_img = copy.deepcopy(self.left_img)
                            with self.right_cam_lock:
                                right_img = copy.deepcopy(self.right_img)
                            with self.chest_cam_lock:
                                chest_img = copy.deepcopy(self.chest_img)
                            with self.depth_cam_lock:
                                depth_raw = copy.deepcopy(self.depth_raw)
                                depth_img = copy.deepcopy(self.depth_img)
                            with self.motor_lock:
                                motor_state = [self._motor_states[i]['angle'] 
                                                for i in range(len(self._motor_states))]
                                print(self.motors)
                                print(motor_state)

                            # Charuco Processing
                            T_bc = get_charuco_camera_pose(chest_img, CHEST_CAM_CAMERA_MTX, CHEST_CAM_DIST_COEF)
                            T_bl = get_charuco_camera_pose(left_img, LEFT_EYE_CAMERA_MTX, LEFT_EYE_DIST_COEF)
                            T_br = get_charuco_camera_pose(right_img, RIGHT_EYE_CAMERA_MTX, RIGHT_EYE_DIST_COEF)

                            if T_bl is not None:
                                l_X = get_world_deproject('left_eye', 
                                                          LEFT_EYE_CAMERA_MTX, 
                                                          LEFT_EYE_DIST_COEF, 
                                                          T_bl)
                                l_x_c, l_y_c, l_z_c = transform_points([l_X], np.linalg.inv(T_bc)).tolist()[0]
                                T_cl, rvec_cl, tvec_cl = chest_rgb_to_eye(T_bc, T_bl)
                            else:
                                l_x_c = -100
                                l_y_c = -100
                                l_z_c = -100
                                rvec_cl = [-100,-100,-100]
                                tvec_cl = [-100,-100,-100]

                            if T_br is not None:
                                r_X = get_world_deproject('right_eye', 
                                                          RIGHT_EYE_CAMERA_MTX, 
                                                          RIGHT_EYE_DIST_COEF, 
                                                          T_br)
                                r_x_c, r_y_c, r_z_c = transform_points([r_X], np.linalg.inv(T_bc)).tolist()[0]
                                T_cr, rvec_cr, tvec_cr = chest_rgb_to_eye(T_bc, T_br)
                            else:
                                r_x_c = -100
                                r_y_c = -100
                                r_z_c = -100
                                rvec_cr = [-100,-100,-100]
                                tvec_cr = [-100,-100,-100]

                            # Storage
                            left_headeyes_dict['x_c'].append(l_x_c)
                            left_headeyes_dict['y_c'].append(l_y_c)
                            left_headeyes_dict['z_c'].append(l_z_c)
                            left_headeyes_dict['cmd_theta_lower_neck_pan'].append(lnp)
                            left_headeyes_dict['cmd_theta_lower_neck_tilt'].append(lnt)
                            left_headeyes_dict['cmd_theta_upper_neck_tilt'].append(unt)
                            left_headeyes_dict['cmd_theta_left_eye'].append(ep)
                            left_headeyes_dict['cmd_theta_tilt'].append(et)
                            left_headeyes_dict['state_theta_lower_neck_pan'].append(motor_state[3])
                            left_headeyes_dict['state_theta_left_lower_neck_tilt'].append(motor_state[6])
                            left_headeyes_dict['state_theta_right_lower_neck_tilt'].append(motor_state[7])
                            left_headeyes_dict['state_theta_left_upper_neck_tilt'].append(motor_state[4])
                            left_headeyes_dict['state_theta_right_upper_neck_tilt'].append(motor_state[5])
                            left_headeyes_dict['state_theta_left_eye'].append(motor_state[0])
                            left_headeyes_dict['state_theta_tilt'].append(motor_state[2])
                            left_headeyes_dict['rvec_0'].append(rvec_cl[0])
                            left_headeyes_dict['rvec_1'].append(rvec_cl[1])
                            left_headeyes_dict['rvec_2'].append(rvec_cl[2])
                            left_headeyes_dict['tvec_0'].append(tvec_cl[0])
                            left_headeyes_dict['tvec_1'].append(tvec_cl[1])
                            left_headeyes_dict['tvec_2'].append(tvec_cl[2])

                            right_headeyes_dict['x_c'].append(r_x_c)
                            right_headeyes_dict['y_c'].append(r_y_c)
                            right_headeyes_dict['z_c'].append(r_z_c)
                            right_headeyes_dict['cmd_theta_lower_neck_pan'].append(lnp)
                            right_headeyes_dict['cmd_theta_lower_neck_tilt'].append(lnt)
                            right_headeyes_dict['cmd_theta_upper_neck_tilt'].append(unt)
                            right_headeyes_dict['cmd_theta_right_eye'].append(ep)
                            right_headeyes_dict['cmd_theta_tilt'].append(et)
                            right_headeyes_dict['state_theta_lower_neck_pan'].append(motor_state[3])
                            right_headeyes_dict['state_theta_left_lower_neck_tilt'].append(motor_state[6])
                            right_headeyes_dict['state_theta_right_lower_neck_tilt'].append(motor_state[7])
                            right_headeyes_dict['state_theta_left_upper_neck_tilt'].append(motor_state[4])
                            right_headeyes_dict['state_theta_right_upper_neck_tilt'].append(motor_state[5])
                            right_headeyes_dict['state_theta_right_eye'].append(motor_state[1])
                            right_headeyes_dict['state_theta_tilt'].append(motor_state[2])
                            right_headeyes_dict['rvec_0'].append(rvec_cr[0])
                            right_headeyes_dict['rvec_1'].append(rvec_cr[1])
                            right_headeyes_dict['rvec_2'].append(rvec_cr[2])
                            right_headeyes_dict['tvec_0'].append(tvec_cr[0])
                            right_headeyes_dict['tvec_1'].append(tvec_cr[1])
                            right_headeyes_dict['tvec_2'].append(tvec_cr[2])

                            # Counter
                            eye_ctr+=1
                            ctr+=1
                            rospy.loginfo("*** Eye Ctr: %i/%i ***" % (eye_ctr, total_eye_ctr))
                            rospy.loginfo("=== Overall Ctr: %i/%i ===" % (ctr, total_ctr))

                            # Visualization
                            concat_img = np.hstack((chest_img, left_img, right_img, depth_img))
                            height, width = concat_img.shape[:2]
                            concat_img = cv2.resize(concat_img, (round(width/2), round(height/2)))

                            # Output Display 1
                            self.rt_display_pub.publish(self.bridge.cv2_to_imgmsg(concat_img, encoding="bgr8"))
        
                    # Pandas Dataframe
                    l_df = pd.DataFrame(left_headeyes_dict)
                    r_df = pd.DataFrame(right_headeyes_dict)

                    # Saving CSV
                    title_str = 'lnt%03i_unt%03i_lnp%03i' % (lnt,unt,lnp)
                    l_csv_path = os.path.join(sub_dir, 'left_'+title_str+'.csv')
                    r_csv_path = os.path.join(sub_dir, 'right_'+title_str+'.csv')
                    l_df.to_csv(l_csv_path, index=False)
                    r_df.to_csv(r_csv_path, index=False)
                    print('Left csv file saved in:', l_csv_path)
                    print('Right csv file saved in:', r_csv_path)

                    # Timer
                    eye_end = time.time()
                    print('[EYE ELAPSED_TIME]', (eye_end-eye_start), 'secs')

        # End
        # Timer
        end = time.time()
        print('[TOTAL ELAPSED_TIME]', (end-self.start)/60.0, 'mins')
        rospy.signal_shutdown('End')
        sys.exit()
    
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
    
    def move_specific(self, names:list, values:list):
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
            img = cv2.line(img, (round(self.camera_mtx[eye]['cx']), 0), (round(self.camera_mtx[eye]['cx']), 480), (0,255,0))
            img = cv2.line(img, (0, round(self.camera_mtx[eye]['cy'])), (848, round(self.camera_mtx[eye]['cy'])), (0,255,0))
            img = cv2.drawMarker(img, (round(self.camera_mtx[eye]['cx']), round(self.camera_mtx[eye]['cy'])), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        else:
            img = cv2.line(img, (round(self.calib_params[eye]['x_center']), 0), (round(self.calib_params[eye]['x_center']), 480), (0,255,0))
            img = cv2.line(img, (0, round(self.calib_params[eye]['y_center'])), (640, round(self.calib_params[eye]['y_center'])), (0,255,0))
            img = cv2.drawMarker(img, (round(self.calib_params[eye]['x_center']), round(self.calib_params[eye]['y_center'])), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        return img


if __name__ == '__main__':
    rospy.init_node('headeyes_sweep')
    vismotor = HeadEyesSweepNode(list_lnt=[0],list_unt=[0])
    vismotor.run()
