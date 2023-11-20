import rospy
import asyncio
from std_msgs.msg import String
from sensor_msgs.msg import Image
import random
import time
from cv_bridge import CvBridge, CvBridgeError
import cv2
import message_filters


class MyNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('my_node')
        self.motors_ready = False
        # Create a publisher
        self.pub = rospy.Publisher('left_eye/processed_example', String, queue_size=1)
        self.pub2 = rospy.Publisher('left_eye/processed_header', String, queue_size=1)

        # Subscribe to the input topic and specify the callback as the message handler
        self.old_stamp = rospy.Time.now()
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber('/left_eye/image_raw', Image, self.pub_callback, queue_size=1)
        self.left_eye_sub = message_filters.Subscriber('/left_eye/image_raw', Image)
        self.sub2 = rospy.Subscriber('left_eye/processed_header', String, self.sub_callback, queue_size=1)
        self.display_r_img_pub = rospy.Publisher('/right_eye/image_processed', Image, queue_size=1)

    def pub_callback(self, msg):
        # Perform processing based on the received message
        # Publish a new message based on the processing result
        delay = random.uniform(0.16, 0.25)
        if random.random() > 0.5:
            delay = 0.0
        time.sleep(delay)
        self.pub2.publish(str(msg.header))

    
    def sub_callback(self, msg):
        # Perform processing based on the received message
        # Publish a new message based on the processing result
        self.motors_ready = True

    def callback(self, msg, msg2):
        # Perform processing based on the received message
        # Publish a new message based on the processing result
        if self.motors_ready:
            fps = self.get_fps(msg.header.stamp)
            self.pub.publish("FPS: %f" % fps)
            left_eye_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Define the text to be added
            text = "FPS: %d" % fps

            # Choose the font type, font scale, and color
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            color = (255, 0, 0)  # BGR color format (here, blue)

            # Determine the text size
            text_size, _ = cv2.getTextSize(text, font, font_scale, 2)

            # Calculate the position to place the text (bottom-left corner)
            text_position = (10, left_eye_img.shape[0] - 10)

            # Add the text to the image
            cv2.putText(left_eye_img, text, text_position, font, font_scale, color, 2)
            self.display_r_img_pub.publish(self.bridge.cv2_to_imgmsg(left_eye_img, encoding="bgr8"))
            cv2.imshow('async', left_eye_img)
            cv2.waitKey(1)
            print('meron')
        self.motors_ready = False

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