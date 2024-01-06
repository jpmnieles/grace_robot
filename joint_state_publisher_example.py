#!/usr/bin/env python

import rospy
import tf
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
import random

def joint_state_publisher():
    # Initialize the ROS node
    rospy.init_node('joint_state_publisher', anonymous=True)

    # Create a publisher for the joint states
    joint_state_pub = rospy.Publisher('joint_states', JointState, queue_size=1)

    # Create a TF broadcaster
    tf_broadcaster = tf.TransformBroadcaster()

    # Set the publishing rate
    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        # Create a JointState message
        joint_state = JointState()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = ['neck_roll', 'neck_pitch', 'neck_yaw',
                            'head_roll', 'head_pitch', 
                            'eyes_pitch', 'lefteye_yaw', 'righteye_yaw']  # Replace with your joint names
        joint_state.position = [random.random(), random.random(), random.random(),
                                random.random(), random.random(),
                                random.random(), random.random(), random.random()]  # Replace with your joint positions
        # joint_state.velocity = [0.0, 0.0, 0.0]  # Replace with your joint velocities

        # Publish the joint state message
        joint_state_pub.publish(joint_state)

        # # Broadcast the TF transforms
        # for i, joint_name in enumerate(joint_state.name):
        #     transform = TransformStamped()
        #     transform.header.stamp = rospy.Time.now()
        #     transform.header.frame_id = 'base_link'  # Replace with your base frame ID
        #     transform.child_frame_id = joint_name
        #     transform.transform.translation.x = 0.0  # Replace with your transform values
        #     transform.transform.translation.y = 0.0  # Replace with your transform values
        #     transform.transform.translation.z = 0.0  # Replace with your transform values
        #     transform.transform.rotation.x = 0.0  # Replace with your transform values
        #     transform.transform.rotation.y = 0.0  # Replace with your transform values
        #     transform.transform.rotation.z = 0.0  # Replace with your transform values
        #     transform.transform.rotation.w = 1.0  # Replace with your transform values

        #     tf_broadcaster.sendTransformMessage(transform)

        # Sleep to maintain the publishing rate
        rate.sleep()

if __name__ == '__main__':
    try:
        joint_state_publisher()
    except rospy.ROSInterruptException:
        pass