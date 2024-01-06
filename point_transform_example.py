import rospy
import tf
from tf.transformations import translation_matrix, quaternion_matrix
from geometry_msgs.msg import PointStamped

import math


def transform_point():
    # Initialize the ROS node
    rospy.init_node('n_transform_point', anonymous=True)

    # Create a TF listener
    tf_listener = tf.TransformListener()
    
    # Source and Target Frame
    source_frame = 'realsense_mount'
    target_frame = 'lefteye'

    # Wait for the TF transform to become available
    tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(1.0))  # Left eye and right are reversed. Observer perspective

    # Create a PointStamped message
    point_source = PointStamped()
    point_source.header.stamp = rospy.Time.now()
    point_source.header.frame_id = source_frame
    point_source.point.x = 1.5
    point_source.point.y = 0.0
    point_source.point.z = 0.0

    (trans,rot) = tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))

    # Convert the translation and rotation to a transformation matrix
    transformation_matrix = translation_matrix(trans)
    rotation_matrix = quaternion_matrix(rot)

    # Apply the transformation to the point
    transformed_point = translation_matrix([point_source.point.x, point_source.point.y, point_source.point.z]) @ transformation_matrix @ rotation_matrix
    print('source:', source_frame)
    print('target:', target_frame)
    new_x = transformed_point[0, 3]
    new_y = transformed_point[1, 3]
    new_z = transformed_point[2, 3]
    print('transformed x:', new_x)
    print('transformed y', new_y)
    print('transformed z', new_z)
    
    # Calculate the pan and tilt angles in radians
    pan = math.atan2(new_y,new_x)
    tilt = math.atan2(new_z,new_x)
    print('new_eye_tilt (rad):', tilt)
    print('new_eye_pan (rad):', pan)
    print('[Note] Tilt first and then pan')
    

if __name__ == '__main__':
    try:
        transform_point()
    except rospy.ROSInterruptException:
        pass