import os
import sys
import math
import time
import yaml
from datetime import datetime

import rospy
from hr_msgs.msg import TargetPosture, MotorStateList
from rospy_message_converter import message_converter
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from grace.utils import *


class ROSMotorClient(object):
    

    def __init__(self, motors=["EyeTurnLeft", "EyeTurnRight", "EyesUpDown"], degrees=True, debug=False):
        self.debug = debug
        self.degrees = degrees
        rospy.init_node('hr_motor_client')
        self.set_motor_names(motors)
        self._motor_state = [None]*self.num_names
        self.motor_sub = rospy.Subscriber('/hr/actuators/motor_states', MotorStateList, self._capture_state)
        self.motor_pub = rospy.Publisher('/hr/actuators/pose', TargetPosture, queue_size=10)
        self.bridge = CvBridge()
        self.left_eye_sub = rospy.Subscriber('/eye_camera/left_eye/image_raw', Image, self._capture_left_image)
        self.right_eye_sub = rospy.Subscriber('/eye_camera/right_eye/image_raw', Image, self._capture_right_image)
        time.sleep(1)
        self.state

    def _capture_left_image(self, msg):
        try:
            self.left_eye_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as error:
            print(error)
    
    def _capture_right_image(self, msg):
        try:
            self.right_eye_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as error:
            print(error)

    def _capture_state(self, msg):
        """Callback for capturing motor state
        """
        self._msg = msg
        for x in msg.motor_states:
            for i, name in enumerate(self.names):
                if x.name == name:
                    self._motor_state[i] = message_converter.convert_ros_message_to_dictionary(x)
                    self._motor_state[i]['angle'] = self._convert_to_angle(name, x.position)
    
    def _convert_to_angle(self, motor, position):
        if self.degrees:
            unit = 360
        else:
            unit = math.pi
        angle = ((position-self._motor_limits[motor]['int_init'])/4096)*unit
        return angle

    def _convert_to_motor_int(self, motor, angle):
        if self.degrees:
            unit = 360
        else:
            unit = math.pi
        angle = round((angle/unit)*4096 + self._motor_limits[motor]['int_init'])
        return angle
        
    def _capture_limits(self, motor):
        int_min = motors_dict[motor]['motor_min']
        int_init = motors_dict[motor]['init']
        int_max = motors_dict[motor]['motor_max']
        deg_min = motor_int_to_angle(motor, int_min, self.degrees)
        deg_init = motor_int_to_angle(motor, int_init, self.degrees)
        deg_max = motor_int_to_angle(motor, int_max, self.degrees)
        limits = {'int_min': int_min, 
                  'int_init': int_init, 
                  'int_max': int_max,
                  'deg_min': deg_min, 
                  'deg_init': deg_init, 
                  'deg_max': deg_max}
        return limits

    def set_motor_names(self, names: list):
        self.names = names
        self.num_names = len(self.names)
        self._motor_limits = {motor: self._capture_limits(motor) for motor in self.names}

    @property
    def state(self):
        return self._motor_state
    
    @property
    def angle_state(self):
        state = []
        for x in self._motor_state:
            state.append(x['angle'])
        return state

    def simple_move(self, values):
        """Commanding move without waiting
        """
        if self.degrees:
            values = [math.radians(x) for x in values]
        args = {"names":self.names, "values":values}
        self.motor_pub.publish(TargetPosture(**args))

    def _check_limits(self, name, value):
        if value < self._motor_limits[name]['deg_min']:
            value = self._motor_limits[name]['deg_min']
        elif value > self._motor_limits[name]['deg_max']:
            value = self._motor_limits[name]['deg_max']
        return value     

    def move(self, values):
        values = [self._check_limits(self.names[i],x) for i,x in enumerate(values)]
        targets = [self._convert_to_motor_int(self.names[i],x) for i,x in enumerate(values)]
        if self.degrees:
            values = [math.radians(x) for x in values]
        args = {"names":self.names, "values":values}
        self.motor_pub.publish(TargetPosture(**args))
        while self._check_target(targets):
            time.sleep(0.02)
            if self.debug:
                print("[DEBUG] Target not yet reached")
            pass
        final_state = self.state
        return final_state
    
    def _check_target(self, targets):
        confirmed = False
        for i in range(self.num_names):
            position = self._motor_state[i]["position"] 
            confirmed |= position > (targets[i]+1) or position  < (targets[i]-1)
            if self.debug:
                print('[DEBUG] name:', self.names[i], 'position:', position, 'target:', targets[i], 'confirmed:',confirmed)
        return confirmed
    
    def get_elapsed_time(self, start_state, end_state):
        start_timestamp = start_state[0]["timestamp"]
        end_timestamp = end_state[-1]["timestamp"]
        elapsed_time = (datetime.fromtimestamp(end_timestamp) 
                        - datetime.fromtimestamp(start_timestamp)).total_seconds()
        return elapsed_time

    def exit(self):
        rospy.signal_shutdown('End of Node')


if __name__ == '__main__':

    # Instantiation
    client = ROSMotorClient(["EyeTurnLeft", "EyeTurnRight", "EyesUpDown"], degrees=True, debug=True)

    # State
    start = time.time()
    state = client.state
    end = time.time()
    print(state)
    print(f"State Elapsed Time: {end-start:.8f} sec")

    # Move with Target
    values = eval(input("Enter the motor commands in list:"))
    start_state = client.state
    end_state = client.move(values)
    elapsed_time = client.get_elapsed_time(start_state, end_state)
    print(f"Move Elapsed Time: {elapsed_time:.8f} sec")
    print(end_state)
    print("State (deg):", client.angle_state)
