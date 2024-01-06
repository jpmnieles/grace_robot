#!/usr/bin/env python

import rospy
import tf
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
import random
import threading


class GraceJointState(object):

    joints_list = ['neck_roll', 'neck_pitch', 'neck_yaw',
                   'head_roll', 'head_pitch', 
                   'eyes_pitch', 'lefteye_yaw', 'righteye_yaw']

    def __init__(self) -> None:
        rospy.init_node('joint_state_publisher', anonymous=True)
        self.var_lock = threading.Lock()
        self.joint_state_pub = rospy.Publisher('joint_states', JointState, queue_size=1)
        self.demand_joint_state_sub = rospy.Subscriber('/demand_joint_states', JointState, self._demand_joint_state)
        self.rate = rospy.Rate(10)  # 10 Hz
        self.names = None
        self.values = None
        self.positions = [0,0,0,0,0,0,0,0]
        rospy.loginfo('Running')
    
    def _demand_joint_state(self, msg):
        with self.var_lock:
            self.names = msg.name
            self.values = msg.position

    def publish_joint_state(self):
        with self.var_lock:
            if self.names is not None or self.values is not None:
                for name, value in zip(self.names, self.values):
                    idx = self.joints_list.index(name)
                    self.positions[idx] = value
        joint_state = JointState()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = self.joints_list
        joint_state.position = self.positions
        self.joint_state_pub.publish(joint_state)

    def run(self):
        while not rospy.is_shutdown():
            # Publish the joint state message
            self.publish_joint_state()

            # Sleep to maintain the publishing rate
            self.rate.sleep()

if __name__ == '__main__':
    grace_joint_state = GraceJointState()
    try:
        grace_joint_state.run()
    except rospy.ROSInterruptException:
        pass