import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import copy

import cv2
from datetime import datetime


rospy.init_node('sync_test')

bridge = CvBridge()

old_ts1 = rospy.Time.now()
old_ts2 = rospy.Time.now()
def get_elapsed_time(ts1, ts2):
    global old_ts1, old_ts2

    left_fps = 1/(ts1-old_ts1).to_sec()
    right_fps = 1/(ts2-old_ts2).to_sec()

    if ts2 > ts1:
        start_timestamp = copy.deepcopy(ts1)
        end_timestamp = copy.deepcopy(ts2)
    else:
        start_timestamp = copy.deepcopy(ts2)
        end_timestamp = copy.deepcopy(ts1)
    elapsed_time = (end_timestamp-start_timestamp).to_sec()
    rospy.loginfo('Left and Right Time Diff: %f sec' % elapsed_time)

    print('Left FPS:', left_fps)
    print('Right FPS:', right_fps)

    old_ts1 = ts1
    old_ts2 = ts2

    return elapsed_time


def gotimage(left_img_msg, right_img_msg):
    left_img = bridge.imgmsg_to_cv2(left_img_msg, "bgr8")
    right_img = bridge.imgmsg_to_cv2(right_img_msg, "bgr8")

    left_stamp = left_img_msg.header.stamp
    right_stamp = right_img_msg.header.stamp
    print(left_img_msg.header, right_img_msg.header)
    get_elapsed_time(left_stamp, right_stamp)

    cv2.imshow('Left Eye', left_img)
    cv2.imshow('Right Eye', right_img)
    cv2.waitKey(1)

left_img_sub = message_filters.Subscriber("/left_eye/image_raw", Image)
right_img_sub = message_filters.Subscriber("/right_eye/image_raw", Image)

ats = message_filters.ApproximateTimeSynchronizer([left_img_sub, right_img_sub], queue_size=1, slop=0.015)
ats.registerCallback(gotimage)
rospy.spin()

