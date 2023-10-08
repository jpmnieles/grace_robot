import rospy
import asyncio
from std_msgs.msg import String
from sensor_msgs.msg import Image
import random
import time

class MyNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('my_node')

        # Create a publisher
        self.pub = rospy.Publisher('left_eye/processed_example', String, queue_size=2)

        # Subscribe to the input topic and specify the callback as the message handler
        self.left_eye_sub = rospy.Subscriber('/left_eye/image_raw', Image, self.callback, queue_size=2)
        self.old_stamp = rospy.Time.now()

    def callback(self, msg):
        # Perform processing based on the received message
        # Publish a new message based on the processing result
        delay = random.uniform(0.1, 1.0)
        time.sleep(delay)
        
        fps = self.get_fps(msg.header.stamp)
        self.pub.publish("FPS: %f" % fps)

    def get_fps(self, new_stamp):
        fps = 1/(new_stamp-self.old_stamp).to_sec()
        self.old_stamp = new_stamp
        return fps

    def run(self):
        # Create an event loop
        loop = asyncio.get_event_loop()

        # Run the event loop until ROS is shut down
        while not rospy.is_shutdown():
            loop.run_forever()

if __name__ == '__main__':
    node = MyNode()
    node.run()