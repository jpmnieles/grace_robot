import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

# Create a CvBridge object to convert ROS messages to OpenCV images
bridge = CvBridge()

# Variables to store the camera intrinsic parameters
fx = None
fy = None
cx = None
cy = None

# Callback function to receive the aligned depth image
def depth_image_callback(msg):
    global fx, fy, cx, cy

    # Convert the ROS depth image message to an OpenCV image
    depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    # Calculate the XYZ values
    points = calculate_xyz_values(depth_image)

    # Print the XYZ values
    for point in points:
        x, y, z = point
        print("X: {:.3f}, Y: {:.3f}, Z: {:.3f}".format(x, y, z))

# Callback function to receive the camera info
def camera_info_callback(msg):
    global fx, fy, cx, cy

    # Retrieve the intrinsic parameters
    fx = msg.K[0]
    fy = msg.K[4]
    cx = msg.K[2]
    cy = msg.K[5]

# Function to calculate the XYZ values from the depth image
def calculate_xyz_values(depth_image):
    # Convert the depth image to a numpy array
    depth_array = np.array(depth_image, dtype=np.float32)

    # Calculate the XYZ values
    points = []
    for y in range(depth_image.shape[0]):
        for x in range(depth_image.shape[1]):
            depth = depth_array[y, x]

            if depth > 0:
                x_normalized = (x - cx) / fx
                y_normalized = (y - cy) / fy

                x_value = depth * x_normalized
                y_value = depth * y_normalized
                z_value = depth

                points.append((x_value, y_value, z_value))

    return points

def main():
    rospy.init_node('realsense_listener')

    # Subscribe to the aligned depth image topic
    rospy.Subscriber('/hr/perception/jetson/realsense/camera/aligned_depth_to_color/image_raw', Image, depth_image_callback)

    # Subscribe to the camera info topic to get the intrinsic parameters
    rospy.Subscriber('/hr/perception/jetson/realsense/camera/aligned_depth_to_color/camera_info', CameraInfo, camera_info_callback)

    rospy.spin()

if __name__ == '__main__':
    main()