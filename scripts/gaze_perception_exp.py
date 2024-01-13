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


class VisuoMotorNode(object):

    rl_state = {  # RL environment on the network takes care of the other side
        'theta_left_pan': None,
        'theta_right_pan': None,
        'theta_tilt': None,
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
        'chest_angle': None,
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
    }

    state_list = {
        'theta_left_pan': [],
        'theta_right_pan': [],
        'theta_tilt': [],
        'chest_cam_px_x': [],
        'chest_cam_px_y': [],  
        'left_eye_px_x': [],
        'left_eye_px_y': [],
        'right_eye_px_x': [],
        'right_eye_px_y': [],
        'dx_l': [],
        'dy_l': [],
        'dx_r': [],
        'dy_r': [],
        '3d_point': [],
        'chest_angle': [],
        'plan_phi_left_pan': [],
        'plan_phi_right_pan': [],
        'plan_phi_tilt': [],
        'chest_img_stamp': [],
        'left_eye_img_stamp': [],
        'right_eye_img_stamp': [],
        'depth_img_stamp': [],
    }

    joints_list = ['neck_roll', 'neck_pitch', 'neck_yaw',
                   'head_roll', 'head_pitch', 
                   'eyes_pitch', 'lefteye_yaw', 'righteye_yaw']
    
    pickle_data = {
        'subject_num': None,
        'markers': [-20, -10, 0, 10, 20],
        'trial_num': None,
        'data': [],
    }
    marker_list = [-20, -10, 0, 10, 20]


    def __init__(self, motors=["EyeTurnLeft", "EyeTurnRight", "EyesUpDown"], degrees=True):
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

        self.joint_state_pub = rospy.Publisher('/demand_joint_states', JointState, queue_size=1)
        self.motor_pub = rospy.Publisher('/hr/actuators/pose', TargetPosture, queue_size=1)
        self.camera_mtx = load_camera_mtx()
        self.bridge = CvBridge()
        time.sleep(1)

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
        self.point_pub = rospy.Publisher('/point_location', PointStamped, queue_size=1)
        self.tf_listener = tf.TransformListener()

        self.key_press_sub = rospy.Subscriber('key_press', String, self._key_press_callback)
        self.internal_ctr = 0
        self.marker = 0
        self.pause = True

        self.chess_idx = 0
        self.ctr = 0
        self.disp_img = np.zeros((480,640,3), dtype=np.uint8)
        self.calib_params = load_json('config/calib/calib_params.json')

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.path.join(os.getcwd(),'pretrained','shape_predictor_68_face_landmarks.dat'))
        dlib.cuda.set_device(0)

        rospy.loginfo('Running')

        self.pickle_data['subject_num'] = input('Enter Subject Number: ')
        self.pickle_data['trial_num'] = input('Enter Trial Number: ')
        self.num_markers = len(self.pickle_data['markers'])
        print('=======Experiment Start=======')
        input("> Press ` to start...\n")

    def _key_press_callback(self, event):
        if event.data == '`':  # Replace '`' with the desired key  # Lowercase tilde (~)
            if self.internal_ctr == 0:
                print('=======(%d deg) =======' % (self.marker_list[self.marker]))
                self.pause = False
                rospy.loginfo('Tracking Enabled')
            elif self.internal_ctr == 1:
                self.pause = True
                rospy.loginfo('Tracking Disabled')
            elif self.internal_ctr == 2:
                rospy.loginfo('Saving data...')

                with self.buffer_lock:
                    self.pickle_data['data'].append(copy.deepcopy(self.rl_state))

                self.internal_ctr = -1
                self.marker+=1
                if self.marker != 5:
                    print('> Move to next marker')
                    # self.pause = False
            
            if self.marker == 5:
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                results_dir = os.path.join(parent_dir, 'results','gaze_perception_subj%d'%(eval(self.pickle_data['subject_num'])))
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                    print(f"Directory created: {results_dir}")
                else:
                    print(f"Directory already exists: {results_dir}")
                
                dt_str = datetime.now().strftime("_%Y%m%d_%H%M%S_%f")
                title_str = 'perception_exp_subj%d_trial%d' % (eval(self.pickle_data['subject_num']), eval(self.pickle_data['trial_num']))
                pickle_path = os.path.join(results_dir, title_str+'_pickle'+dt_str+'.pickle')

                # Saving to Pickle File
                with open(pickle_path, 'wb') as file:
                    pickle.dump(self.pickle_data, file)
                print('=====================')
                print('Pickle file saved in:', pickle_path)

                # Saving CSV
                df = pd.DataFrame(self.state_list)

                # Save DataFrame as CSV
                csv_path = os.path.join(results_dir, title_str+'_csv'+dt_str+'.csv')
                df.to_csv(csv_path, index=False)
                print('CSV file saved:',csv_path)

                rospy.loginfo("Press 'Enter' to shutdown")
                rospy.signal_shutdown('End')
                sys.exit()

            self.internal_ctr+=1

    def set_action(self, action):
        with self.action_lock:
            self.action = action

    def publish_joint_state(self, names, values):
        joint_state = JointState()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = names
        joint_state.position = values
        self.joint_state_pub.publish(joint_state)

    def _capture_state(self, msg):
        """Callback for capturing motor state
        """
        eye_motors_list = []
        self._msg = msg
        for idx, x in enumerate(msg.motor_states):
            if x.name in self.names:
                eye_motors_list.append(idx)
        if len(eye_motors_list) == 3:
            # rospy.loginfo('Complete')
            # rospy.loginfo(msg.motor_states[eye_motors_list[0]])
            # rospy.loginfo(msg.motor_states[eye_motors_list[1]])
            # rospy.loginfo(msg.motor_states[eye_motors_list[2]])
            with self.motor_lock:
                for i in range(self.num_names):
                    motor_msg = msg.motor_states[eye_motors_list[i]]
                    idx = self.motors.index(motor_msg.name)
                    self._motor_states[idx] = message_converter.convert_ros_message_to_dictionary(motor_msg)
                    self._motor_states[idx]['angle'] = motor_int_to_angle(motor_msg.name, motor_msg.position, self.degrees)
        else:
            # rospy.loginfo('Incomplete')
            pass

    def depth_to_pointcloud(self, px, depth_img, z_replace=1.0):
        fx = self.camera_mtx['chest_cam']['fx']
        cx = self.camera_mtx['chest_cam']['cx']
        fy = self.camera_mtx['chest_cam']['fy']
        cy = self.camera_mtx['chest_cam']['cy']
        cx = 434
        cy = 218
        u = round(px[0])
        v = round(px[1])
        z = depth_img[v,u]/1000.0
        if z==0:
            z = z_replace
        x = ((u-cx)/fx)*z
        y = ((v-cy)/fy)*z
        
        # Intel Realsense Camera Link
        pts = [x,y,z]
        (trans,rot) = self.tf_listener.lookupTransform('camera_link', 'camera_aligned_depth_to_color_frame', rospy.Time(0))
        transformation_matrix = translation_matrix(trans)
        rotation_matrix = quaternion_matrix(rot)
        transformed_point = translation_matrix(pts) @ transformation_matrix @ rotation_matrix
        new_x = transformed_point[0, 3]
        new_y = transformed_point[1, 3]
        new_z = transformed_point[2, 3]
        return (new_x,new_y,new_z)
    
    def transform_point(self, source_frame, target_frame, pts:list):
        (trans,rot) = self.tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
        transformation_matrix = translation_matrix(trans)
        rotation_matrix = quaternion_matrix(rot)
        transformed_point = translation_matrix(pts) @ transformation_matrix @ rotation_matrix
        new_x = transformed_point[0, 3]
        new_y = transformed_point[1, 3]
        new_z = transformed_point[2, 3]
        return (new_x, new_y, new_z)

    def eye_imgs_callback(self, left_img_msg, right_img_msg, chest_img_msg, depth_img_msg):
        # Motor Trigger Sync (3.33 FPS or 299.99 ms)
        max_stamp = max(left_img_msg.header.stamp, right_img_msg.header.stamp, chest_img_msg.header.stamp)
        elapsed_time = (max_stamp - self.frame_stamp_tminus1).to_sec()
        if elapsed_time > 285e-3:
            start = time.time()
            # print('--------------')
            # rospy.loginfo(f'FPS: {1/elapsed_time: .{2}f}')
            self.frame_stamp_tminus1 = max_stamp

            # Initialization
            dx_l, dy_l, dx_r, dy_r = 0, 0, 0, 0
            theta_l_pan, theta_r_pan = None, None
            theta_l_tilt, theta_r_tilt, theta_tilt = None, None, None

            # Conversion of ROS Message
            self.left_img = self.bridge.imgmsg_to_cv2(left_img_msg, "bgr8")
            self.right_img = self.bridge.imgmsg_to_cv2(right_img_msg, "bgr8")
            self.chest_img = self.bridge.imgmsg_to_cv2(chest_img_msg, "bgr8")
            self.depth_img = self.bridge.imgmsg_to_cv2(depth_img_msg, "16UC1")
            # print(left_img_msg.header, right_img_msg.header, chest_img_msg.header)
            # print(self.depth_img)

            ## Attention ##
            
            # Face Detection
            self.attention.register_imgs(self.left_img, self.right_img)
            self.attention.detect_people(self.left_img, self.right_img)
            
            # Attention
            id = 0  # Person ID
            if len(self.attention.l_detections) > 0:
                dx_l, dy_l = self.attention.get_pixel_target(id, 'left_eye')

            if len(self.attention.r_detections) > 0:
                dx_r, dy_r = self.attention.get_pixel_target(id, 'right_eye')
            
            # Process Left Eye, Right Eye, Chest Cam Target
            l_gray = cv2.cvtColor(copy.deepcopy(self.left_img), cv2.COLOR_BGR2GRAY)
            r_gray = cv2.cvtColor(copy.deepcopy(self.right_img), cv2.COLOR_BGR2GRAY)
            c_gray = cv2.cvtColor(copy.deepcopy(self.chest_img), cv2.COLOR_BGR2GRAY)

            l_detections = self.detector(l_gray, 0)
            r_detections = self.detector(r_gray, 0)
            c_detections = self.detector(c_gray, 0)
            if len(c_detections) > 0:
                c_detection = c_detections[0]
                landmarks = self.predictor(c_gray, c_detection)
                x_target = landmarks.part(28).x
                y_target = landmarks.part(28).y
                chest_cam_px = (x_target, y_target)
                chest_img = cv2.rectangle(copy.deepcopy(self.chest_img), (c_detection.left(), c_detection.top()), 
                              (c_detection.right(), c_detection.bottom()), (0, 0, 255), 2)
                chest_img = cv2.drawMarker(chest_img, (round(x_target),round(y_target)), color=(255, 0, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=13, thickness=2)
            else:
                chest_cam_px = (424, 240)
                chest_img = self.chest_img

            if len(l_detections) > 0:
                l_detection = l_detections[0]
                landmarks = self.predictor(l_gray, l_detection)
                x_target = landmarks.part(28).x
                y_target = landmarks.part(28).y
                left_eye_px = (x_target, y_target)
                left_img = cv2.rectangle(copy.deepcopy(self.left_img), (l_detection.left(), l_detection.top()), 
                              (l_detection.right(), l_detection.bottom()), (0, 0, 255), 2)
                left_img = cv2.drawMarker(left_img, (round(x_target),round(y_target)), color=(255, 0, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=13, thickness=2)
            else:
                left_eye_px = (-4, -4)
                left_img = self.left_img
            
            if len(r_detections) > 0:
                r_detection = r_detections[0]
                landmarks = self.predictor(r_gray, r_detection)
                x_target = landmarks.part(28).x
                y_target = landmarks.part(28).y
                right_eye_px = (x_target, y_target)
                right_img = cv2.rectangle(copy.deepcopy(self.right_img), (r_detection.left(), r_detection.top()), 
                              (r_detection.right(), r_detection.bottom()), (0, 0, 255), 2)
                right_img = cv2.drawMarker(right_img, (round(x_target),round(y_target)), color=(255, 0, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=13, thickness=2)
            else:
                right_eye_px = (-4, -4)
                right_img = self.right_img


            ## Geometric Intersection
            # Point Stamp
            point_msg = PointStamped()
            point_msg.header.stamp = rospy.Time.now()
            point_msg.header.frame_id = 'realsense_mount'  # Replace with your desired frame ID
            x,y,z = self.depth_to_pointcloud(chest_cam_px, self.depth_img, 1.0)
            
            # x (straight away from robot, depth), y (positive left, negative right), z (negative down, position right)
            target_x = max(0.3, z)
            target_y = -x
            target_z = -y
            point_msg.point.x = target_x
            point_msg.point.y = target_y
            point_msg.point.z = target_z
            # print("Chest_cam_px", chest_cam_px)
            # print("Point", target_x, target_y, target_z)
            # Publish
            self.point_pub.publish(point_msg)

            # Angles Calculation
            left_eye_pts = self.transform_point(source_frame='realsense_mount', target_frame='lefteye',
                                                pts=[target_x, target_y, target_z])
            right_eye_pts = self.transform_point(source_frame='realsense_mount', target_frame='righteye',
                                                pts=[target_x, target_y, target_z])
            cyclops_eye_pts = self.transform_point(source_frame='realsense_mount', target_frame='eyes',
                                                pts=[target_x, target_y, target_z])
            eyes_tilt = math.atan2(cyclops_eye_pts[2], cyclops_eye_pts[0])
            left_pan = math.atan2(left_eye_pts[1], left_eye_pts[0])
            right_pan = math.atan2(right_eye_pts[1], right_eye_pts[0])
            # rospy.loginfo(f"left_eye_pan (rad): {left_pan: .{4}f}")
            # rospy.loginfo(f"right_eye_pan (rad): {right_pan: .{4}f}")
            # rospy.loginfo(f"eyes_tilt (rad): {eyes_tilt: .{4}f}")
            
            # Publish Joint States
            if target_x != 0.3:
                joints = ['eyes_pitch']
                positions = [eyes_tilt]
                self.publish_joint_state(joints, positions)
                joints = ['lefteye_yaw', 'righteye_yaw']
                positions = [left_pan, right_pan]
                self.publish_joint_state(joints, positions)
                
                # Output of the Geometric Intersection
                theta_l_pan = math.degrees(-left_pan)/self.calib_params['left_eye']['slope']
                theta_r_pan = math.degrees(-right_pan)/self.calib_params['right_eye']['slope']
                theta_tilt = math.degrees(eyes_tilt)/self.calib_params['tilt_eyes']['slope']

            # Storing
            with self.buffer_lock:
                # Get Motor State
                with self.motor_lock:
                    self.calibration.store_latest_state(self._motor_states)
                
                    # Calibration Algorithm
                    theta_l_pan, theta_l_tilt = self.calibration.compute_left_eye_cmd(dx_l, dy_l)
                    theta_r_pan, theta_r_tilt = self.calibration.compute_right_eye_cmd(dx_r, dy_r) 
                    theta_tilt = self.calibration.compute_tilt_cmd(theta_l_tilt, theta_r_tilt, alpha_tilt=0.5)
                    self.calibration.store_cmd(theta_l_pan, theta_r_pan, theta_tilt)

                    self.rl_state['theta_left_pan'] = self._motor_states[0]['angle']
                    self.rl_state['theta_right_pan'] = self._motor_states[1]['angle']
                    self.rl_state['theta_tilt'] = self._motor_states[2]['angle']
                    self.rl_state['chest_cam_px_x'] = chest_cam_px[0]
                    self.rl_state['chest_cam_px_y'] = chest_cam_px[1]
                    self.rl_state['left_eye_px_x'] = left_eye_px[0]
                    self.rl_state['left_eye_px_y'] = left_eye_px[1]
                    self.rl_state['right_eye_px_x'] = right_eye_px[0]
                    self.rl_state['right_eye_px_y'] = right_eye_px[1]
                    self.rl_state['dx_l'] = dx_l
                    self.rl_state['dy_l'] = dy_l
                    self.rl_state['dx_r'] = dx_r
                    self.rl_state['dy_r'] = dy_r 

                    self.rl_state['3d_point'] = (target_x, target_y, target_z)
                    self.rl_state['chest_angle'] =  math.atan2(target_y, target_x)
                    
                    self.rl_state['plan_phi_left_pan'] = -left_pan
                    self.rl_state['plan_phi_right_pan'] = -right_pan
                    self.rl_state['plan_phi_tilt'] = eyes_tilt

                    self.rl_state['chest_img'] = self.chest_img
                    self.rl_state['left_eye_img'] = self.left_img
                    self.rl_state['right_eye_img'] = self.right_img
                    self.rl_state['depth_img'] = self.depth_img

                    self.rl_state['chest_img_stamp'] = chest_img_msg.header.stamp.to_sec()
                    self.rl_state['left_eye_img_stamp'] = left_img_msg.header.stamp.to_sec()
                    self.rl_state['right_eye_img_stamp'] = right_img_msg.header.stamp.to_sec()
                    self.rl_state['depth_img_stamp'] = depth_img_msg.header.stamp.to_sec()
                    # print(self.rl_state)

                    # CSV Save
                    self.state_list['theta_left_pan'].append(self._motor_states[0]['angle'])
                    self.state_list['theta_right_pan'].append(self._motor_states[1]['angle'])
                    self.state_list['theta_tilt'].append(self._motor_states[2]['angle'])
                    self.state_list['chest_cam_px_x'].append(chest_cam_px[0])
                    self.state_list['chest_cam_px_y'].append(chest_cam_px[1])
                    self.state_list['left_eye_px_x'].append(left_eye_px[0])
                    self.state_list['left_eye_px_y'].append(left_eye_px[1])
                    self.state_list['right_eye_px_x'].append(right_eye_px[0])
                    self.state_list['right_eye_px_y'].append(right_eye_px[1])

                    self.state_list['dx_l'].append(dx_l)
                    self.state_list['dy_l'].append(dy_l)
                    self.state_list['dx_r'].append(dx_r)
                    self.state_list['dy_r'].append(dy_r)

                    self.state_list['3d_point'].append((target_x, target_y, target_z))
                    self.state_list['chest_angle'].append(math.atan2(target_y, target_x))

                    self.state_list['plan_phi_left_pan'].append(-left_pan)
                    self.state_list['plan_phi_right_pan'].append(-right_pan)
                    self.state_list['plan_phi_tilt'].append(eyes_tilt)

                    self.state_list['chest_img_stamp'].append(chest_img_msg.header.stamp.to_sec())
                    self.state_list['left_eye_img_stamp'].append(left_img_msg.header.stamp.to_sec())
                    self.state_list['right_eye_img_stamp'].append(right_img_msg.header.stamp.to_sec())
                    self.state_list['depth_img_stamp'].append(depth_img_msg.header.stamp.to_sec())

            # Movement
            # theta_l_pan, theta_r_pan, theta_tilt = None, None, None
            end = time.time()
            # print('[ELAPSED_TIME]', (end-start)*1000, 'msecs')
            # Wait for the new command

            with self.action_lock:
                if self.action != None:
                    theta_l_pan = self.action[0]
                    theta_r_pan = self.action[1]
                    theta_tilt = self.action[2]

            if ((theta_l_pan is not None) or (theta_r_pan is not None) or (theta_tilt is not None)) and (not self.pause):
                self.move((theta_l_pan, theta_r_pan, theta_tilt))

            # Visualization
            left_img = self.ctr_cross_img(left_img, 'left_eye')
            right_img = self.ctr_cross_img(right_img, 'right_eye')
            chest_img = self.ctr_cross_img(chest_img, 'chest_cam')
            concat_img = np.hstack((chest_img, left_img, right_img))
            
            # Resizing
            height, width = concat_img.shape[:2]
            concat_img = cv2.resize(concat_img, (round(width/2), round(height/2)))

            # Output Display 1
            self.rt_display_pub.publish(self.bridge.cv2_to_imgmsg(concat_img, encoding="bgr8"))

            # Reassigning
            self.chess_idx_tminus1 = self.chess_idx
    
            # # Saving
            # self.ctr+=1
            # if self.ctr==self.num_ctr:
            #     # Create a DataFrame
            #     df = pd.DataFrame(self.state_list)

            #     # Save DataFrame as CSV
            #     parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            #     results_dir = os.path.join(parent_dir, 'results')
            #     dt_str = datetime.now().strftime("_%Y%m%d_%H%M%S_%f")
            #     title_str = 'acc_baseline'
            #     fn_path = os.path.join(results_dir, title_str+dt_str+'.csv')
            #     df.to_csv(fn_path, index=False)
            #     print('File saved:', fn_path)
                
            #     rospy.signal_shutdown('End of program')
            #     sys.exit()

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

    def ctr_cross_img(self, img, eye:str):
        """eye (str): select from ['left_eye', 'right_eye', 'chest_cam']
        """
        if eye == 'chest_cam':
            # True Optical Center
            img = cv2.line(img, (round(self.camera_mtx[eye]['cx']), 0), (round(self.camera_mtx[eye]['cx']), 480), (0,255,0))
            img = cv2.line(img, (0, round(self.camera_mtx[eye]['cy'])), (848, round(self.camera_mtx[eye]['cy'])), (0,255,0))
            img = cv2.drawMarker(img, (round(self.camera_mtx[eye]['cx']), round(self.camera_mtx[eye]['cy'])), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        
            # Adjusted Center
            img = cv2.line(img, (434, 0), (434, 480), (255,0,0))
            img = cv2.line(img, (0, 218), (848, 218), (255,0,0))
            img = cv2.drawMarker(img, (434, 218), color=(255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        else:
            img = cv2.line(img, (round(self.calib_params[eye]['x_center']), 0), (round(self.calib_params[eye]['x_center']), 480), (0,255,0))
            img = cv2.line(img, (0, round(self.calib_params[eye]['y_center'])), (640, round(self.calib_params[eye]['y_center'])), (0,255,0))
            img = cv2.drawMarker(img, (round(self.calib_params[eye]['x_center']), round(self.calib_params[eye]['y_center'])), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        return img


if __name__ == '__main__':
    rospy.init_node('visuomotor')
    vismotor = VisuoMotorNode()
    rospy.spin()
