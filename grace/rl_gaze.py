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
import datetime
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
        'chest_cam_px': None, 
        'left_eye_px': None,
        'right_eye_px': None,
        'plan_phi_left_pan': None,
        'plan_phi_right_pan': None,
        'plan_phi_tilt': None,
        'chest_img_stamp': None,
        'left_eye_img_stamp': None,
        'right_eye_img_stamp': None,
    }

    joints_list = ['neck_roll', 'neck_pitch', 'neck_yaw',
                   'head_roll', 'head_pitch', 
                   'eyes_pitch', 'lefteye_yaw', 'righteye_yaw']


    def __init__(self, motors=["EyeTurnLeft", "EyeTurnRight", "EyesUpDown"], degrees=True):
        self.motor_lock = threading.Lock()
        self.buffer_lock = threading.Lock()
        self.action_lock = threading.Lock()
        self.chess_idx_lock = threading.Lock()
        self.motors = motors
        self.degrees = degrees
        self._set_motor_limits(motors)

        self.attention = ChessboardAttention()
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
        self.state_pub = rospy.Publisher('/grace/state', String, queue_size=1)
        self.point_pub = rospy.Publisher('/point_location', PointStamped, queue_size=1)
        self.tf_listener = tf.TransformListener()

        self.chess_idx = 0
        self.ctr = 0
        self.disp_img = np.zeros((480,640,3), dtype=np.uint8)
        self.calib_params = load_json('config/calib/calib_params.json')
        rospy.loginfo('Running')

    def set_chess_idx(self, chess_idx):
        with self.chess_idx_lock:
            self.chess_idx = chess_idx

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

    def depth_to_pointcloud(self, px, depth_img, z_replace=1.5):
        fx = self.camera_mtx['chest_cam']['fx']
        cx = self.camera_mtx['chest_cam']['cx']
        fy = self.camera_mtx['chest_cam']['fy']
        cy = self.camera_mtx['chest_cam']['cy']
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
            
            # Random Target
            # chess_idx = random.randint(0,17)
            # self.set_chess_idx(chess_idx)
            
            # For calibration
            # self.chess_idx = 7

            # Sequential  
            # if self.ctr%2 == 0: 
            #     if self.chess_idx == 17:
            #         self.chess_idx = 0
            #         self.ctr = -1
            #     else:
            #         self.chess_idx += 1
            # self.ctr+=1
            
            # Process Left Eye, Right Eye, Chest Cam Target
            left_eye_pxs = self.attention.process_img(self.chess_idx, self.left_img)
            right_eye_pxs = self.attention.process_img(self.chess_idx, self.right_img)
            chest_cam_pxs = self.attention.process_img(self.chess_idx, self.chest_img)

            # Calculate Delta between Gaze Center and Pixel Target
            if chest_cam_pxs is None:
                left_eye_px = (-self.calib_params['left_eye']['x_center'], -self.calib_params['left_eye']['y_center'])
                right_eye_px = (-self.calib_params['right_eye']['x_center'], -self.calib_params['right_eye']['y_center'])
                chest_cam_px = (424, 240)
                left_eye_px_tminus1 = left_eye_px
                right_eye_px_tminus1 = right_eye_px
                chest_cam_px_tminus1 = chest_cam_px
            elif left_eye_pxs is None or right_eye_pxs is None:
                dx_l, dy_l, dx_r, dy_r = 0, 0, 0, 0
                left_eye_px = (-self.calib_params['left_eye']['x_center'], -self.calib_params['left_eye']['y_center'])
                right_eye_px = (-self.calib_params['right_eye']['x_center'], -self.calib_params['right_eye']['y_center'])
                chest_cam_px = tuple(chest_cam_pxs[self.chess_idx].tolist())
                left_eye_px_tminus1 = left_eye_px
                right_eye_px_tminus1 = right_eye_px
                chest_cam_px_tminus1 = chest_cam_pxs[self.chess_idx_tminus1]
            else:
                # Preprocessing
                left_eye_px = tuple(left_eye_pxs[self.chess_idx].tolist())
                right_eye_px = tuple(right_eye_pxs[self.chess_idx].tolist())
                chest_cam_px = tuple(chest_cam_pxs[self.chess_idx].tolist())
                left_eye_px_tminus1 = left_eye_pxs[self.chess_idx_tminus1]
                right_eye_px_tminus1 = right_eye_pxs[self.chess_idx_tminus1]
                chest_cam_px_tminus1 = chest_cam_pxs[self.chess_idx_tminus1]


            ## Geometric Intersection
            # Point Stamp
            point_msg = PointStamped()
            point_msg.header.stamp = rospy.Time.now()
            point_msg.header.frame_id = 'realsense_mount'  # Replace with your desired frame ID
            x,y,z = self.depth_to_pointcloud(chest_cam_px, self.depth_img, 1.52)
            
            # x (straight away from robot, depth), y (positive left, negative right), z (negative down, position right)
            y_offset = 0.328
            y_offset = 0
            z_offset = 0
            target_x = max(0.3, z)
            target_y = -x + y_offset
            target_z = -y + z_offset
            point_msg.point.x = target_x
            point_msg.point.y = target_y
            point_msg.point.z = target_z
            print("Chest_cam_px", chest_cam_px)
            print("Point", target_x, target_y, target_z)
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

            # Visualize the Previous Target
            left_img = cv2.drawMarker(self.left_img, (round(left_eye_px[0]),round(left_eye_px[1])), color=(204, 41, 204), 
                                markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)
            left_img = cv2.drawMarker(left_img, (round(left_eye_px_tminus1[0]),round(left_eye_px_tminus1[1])), color=(0, 0, 255), 
                                markerType=cv2.MARKER_TILTED_CROSS, markerSize=13, thickness=2)
            right_img = cv2.drawMarker(self.right_img, (round(right_eye_px[0]),round(right_eye_px[1])), color=(204, 41, 204), 
                        markerType=cv2.MARKER_STAR, markerSize=13, thickness=2)
            right_img = cv2.drawMarker(right_img, (round(right_eye_px_tminus1[0]),round(right_eye_px_tminus1[1])), color=(0, 0, 255), 
                        markerType=cv2.MARKER_TILTED_CROSS, markerSize=13, thickness=2)
            chest_img = cv2.drawMarker(self.chest_img, (round(chest_cam_px[0]),round(chest_cam_px[1])), color=(204, 41, 204), 
                        markerType=cv2.MARKER_STAR, markerSize=13, thickness=2)
            chest_img = cv2.drawMarker(chest_img, (round(chest_cam_px_tminus1[0]),round(chest_cam_px_tminus1[1])), color=(0, 0, 255), 
                        markerType=cv2.MARKER_TILTED_CROSS, markerSize=13, thickness=2)

            # Storing
            with self.buffer_lock:
                # Get Motor State
                with self.motor_lock:
                    self.calibration.store_latest_state(self._motor_states)

                    self.rl_state['chest_cam_px'] = chest_cam_px
                    self.rl_state['left_eye_px'] = left_eye_px
                    self.rl_state['right_eye_px'] = right_eye_px
                    self.rl_state['theta_left_pan'] = self._motor_states[0]['angle']
                    self.rl_state['theta_right_pan'] = self._motor_states[1]['angle']
                    self.rl_state['theta_tilt'] = self._motor_states[2]['angle']
                    self.rl_state['plan_phi_left_pan'] = -left_pan
                    self.rl_state['plan_phi_right_pan'] = -right_pan
                    self.rl_state['plan_phi_tilt'] = eyes_tilt
                    self.rl_state['chest_img_stamp'] = chest_img_msg.header.stamp.to_sec()
                    self.rl_state['left_eye_img_stamp'] = left_img_msg.header.stamp.to_sec()
                    self.rl_state['right_eye_img_stamp'] = right_img_msg.header.stamp.to_sec()
                    # print(self.rl_state)

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

            if (theta_l_pan is not None) or (theta_r_pan is not None) or (theta_tilt is not None):
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

            # Publish States
            json_str = json.dumps(self.rl_state)
            message = String()
            message.data = json_str

            # Publish the message
            self.state_pub.publish(message)

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
