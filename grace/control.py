import os
import sys
import math
import time
import yaml
from datetime import datetime
import threading

import rospy
from hr_msgs.msg import TargetPosture, MotorStateList
from rospy_message_converter import message_converter
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from grace.utils import *


class ROSMotorClient(object):
    

    def __init__(self, motors=["EyeTurnLeft", "EyeTurnRight", "EyesUpDown"], degrees=True, debug=False):
        self.motor_lock = threading.Lock()
        self.debug = debug
        self.degrees = degrees
        rospy.init_node('hr_motor_client')
        self.set_motor_names(motors)
        self._motor_state = [None]*self.num_names
        self.motor_sub = rospy.Subscriber('/hr/actuators/motor_states', MotorStateList, self._capture_state)
        self.motor_pub = rospy.Publisher('/hr/actuators/pose', TargetPosture, queue_size=10)
        self.bridge = CvBridge()
        self.left_eye_sub = rospy.Subscriber('/left_eye/image_raw', Image, self._capture_left_image)
        self.right_eye_sub = rospy.Subscriber('/right_eye/image_raw', Image, self._capture_right_image)
        time.sleep(1)
        self.state

    def _capture_left_image(self, msg):
        try:
            self.left_eye_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.left_timestamp = msg.header.stamp.to_time()
        except CvBridgeError as error:
            print(error)
    
    def _capture_right_image(self, msg):
        try:
            self.right_eye_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.right_timestamp = msg.header.stamp.to_time()
        except CvBridgeError as error:
            print(error)

    def _capture_state(self, msg):
        """Callback for capturing motor state
        """
        with self.motor_lock:
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

    def set_motor_names(self, names: list):
        self.names = names
        self.num_names = len(self.names)
        self._motor_limits = {motor: self._capture_limits(motor) for motor in self.names}

    @property
    def state(self):
        with self.motor_lock:
            state = self._motor_state.copy()
        return state
    
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
        if value < self._motor_limits[name]['angle_min']:
            value = self._motor_limits[name]['angle_min']
        elif value > self._motor_limits[name]['angle_max']:
            value = self._motor_limits[name]['angle_max']
        return value     

    def move(self, values):
        values = [self._check_limits(self.names[i],x) for i,x in enumerate(values)]
        if self.degrees:
            values = [math.radians(x) for x in values]
        args = {"names":self.names, "values":values}
        self.motor_pub.publish(TargetPosture(**args))
        final_state = self._motor_state
        return final_state
    
    def _compute_delay(self, cmd, tol=0.1):
        # Units is degrees
        if cmd <= 20-tol:
            delay = 165e-3
        elif (cmd > 20-tol) and (cmd <= 45-tol):
            delay = 198e-3
        elif cmd > 45-tol:
            delay = 231e-3
        return delay
    
    def get_elapsed_time(self, start_state, end_state):
        start_timestamp = start_state[0]["timestamp"]
        end_timestamp = end_state[0]["timestamp"]
        elapsed_time = (datetime.fromtimestamp(end_timestamp)-datetime.fromtimestamp(start_timestamp)).total_seconds()
        return elapsed_time
    
    def slow_move(self, idx, position, step_size, time_interval, num_repeat=1):  # TODO: Only limited to only 1 motor only. Extend to multimotor
        state = self.state
        curr_position_list = [state[idx]['angle'] for idx in range(self.num_names)]
        curr_position = curr_position_list[idx]
        if position > curr_position:
            sign = 1
        else:
            sign = -1
        target_wave = generate_target_wave(target_amp=position, init_amp=curr_position, step_size=sign*step_size, num_cycles=num_repeat)
        for cmd in target_wave:
            time.sleep(time_interval)
            curr_position_list[idx] = cmd
            self.move(curr_position_list)
        time.sleep(0.3333)

    def exit(self):
        rospy.signal_shutdown('End of Node')


if __name__ == '__main__':

    # Instantiation
    client = ROSMotorClient(["EyeTurnLeft", "EyeTurnRight", "EyesUpDown"], degrees=True, debug=False)

    # State
    start = time.time()
    state = client.state
    end = time.time()
    print(state)
    print(f"State Elapsed Time: {end-start:.8f} sec")

    # Move with Target
    values = eval(input("Enter the motor commands in list:"))
    start_state = client.state
    client.move(values)
    time.sleep(0.3333)
    end_state = client.state
    elapsed_time = client.get_elapsed_time(start_state, end_state)
    print(f"Move Elapsed Time: {elapsed_time:.8f} sec")
    print(end_state)
    print("State (deg):", client.angle_state)

    # Slow Move Target
    value = eval(input("Enter the motor position: "))
    client.slow_move(idx=0,position=value,step_size=0.0879,time_interval=0.015)
