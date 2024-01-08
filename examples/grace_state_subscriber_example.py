import rospy
from std_msgs.msg import String
import json

def json_callback(data):
    # Parse the JSON data
    json_data = json.loads(data.data)

    # Process or use the JSON data
    print("Received JSON data:")
    print(json_data)
    # ... process or use more fields as needed

def json_subscriber():
    # Initialize the ROS node
    rospy.init_node('json_subscriber', anonymous=True)

    # Create a subscriber for the JSON topic
    rospy.Subscriber('/grace/state', String, json_callback)

    # Start the ROS spin loop
    rospy.spin()

if __name__ == '__main__':
    json_subscriber()