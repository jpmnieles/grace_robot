import rospy
from geometry_msgs.msg import PointStamped

def point_stamped_publisher():
    # Initialize the ROS node
    rospy.init_node('point_stamped_publisher', anonymous=True)

    # Create a publisher for the PointStamped message
    point_pub = rospy.Publisher('point_stamped', PointStamped, queue_size=1)

    rospy.loginfo('Running')

    # Set the publishing rate
    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        # Create a PointStamped message
        point_stamped = PointStamped()
        point_stamped.header.stamp = rospy.Time.now()
        point_stamped.header.frame_id = 'realsense_mount'  # Replace with your desired frame ID
        point_stamped.point.x = 1.5  # Replace with your point x-coordinate
        point_stamped.point.y = 0  # Replace with your point y-coordinate
        point_stamped.point.z = 0  # Replace with your point z-coordinate

        # Publish the PointStamped message
        point_pub.publish(point_stamped)

        # Sleep to maintain the publishing rate
        rate.sleep()

if __name__ == '__main__':
    try:
        point_stamped_publisher()
    except rospy.ROSInterruptException:
        pass