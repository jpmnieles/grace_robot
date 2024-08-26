import os
import sys
sys.path.append(os.getcwd())

import math
import time
import threading

import rospy
from hr_msgs.msg import TargetPosture, MotorStateList
from rospy_message_converter import message_converter

from grace.utils import *


class SimpleNode(object):


    def __init__(self, motors=["EyeTurnLeft", "EyeTurnRight", "EyesUpDown",
                         "NeckRotation", "UpperGimbalLeft", "UpperGimbalRight",
                         "LowerGimbalLeft", "LowerGimbalRight"], degrees=True):

        rospy.loginfo('Starting')
        self.motors = motors
        self._motor_states = [None]*len(self.motors)
        self.degrees = degrees
        self.motor_lock = threading.Lock()
        
        self.motor_pub = rospy.Publisher('/hr/actuators/pose', TargetPosture, queue_size=1)
        self.motor_sub = rospy.Subscriber('/hr/actuators/motor_states', MotorStateList, self._capture_state)
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
      
    def main(self):
        rospy.loginfo('Running')

        rospy.sleep(1.5)
        self.move_specific(["UpperGimbalLeft","UpperGimbalRight","LowerGimbalLeft","LowerGimbalRight"],
                                           [44,-44,-13,13])
        rospy.sleep(1.5)

        rospy.signal_shutdown('End')
        sys.exit()
    
    
    def move_specific(self, names:list, values:list):
        if self.degrees:
            values = [math.radians(x) for x in values]
        args = {"names":names, "values":values}
        self.motor_pub.publish(TargetPosture(**args))


if __name__ == '__main__':
    rospy.init_node('SimpleNode')
    simple = SimpleNode()
    simple.main()
