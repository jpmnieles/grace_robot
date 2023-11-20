import rospy
from hr_msgs.msg import Target
import math
from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import Image, CameraInfo
import tf


rospy.init_node("gaze_animation")
head_focus_pub = rospy.Publisher('/hr/animation/set_face_target', Target, queue_size=1)
gaze_focus_pub = rospy.Publisher('/hr/animation/set_gaze_target', Target, queue_size=1)
tf_listener = tf.TransformListener(False, rospy.Duration.from_sec(1))


class PosStruct():

    def __init__(self) -> None:
        self.x = 0
        self.y = 0
        self.z = 0
    
    def set_xyz(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    

def get_blender_pos(pos, ts, frame_id):
    if frame_id == 'blender':
        return pos
    else:
        ps = PointStamped()
        ps.header.seq = 0
        ps.header.stamp = ts
        ps.header.frame_id = frame_id
        ps.point.x = pos.x
        ps.point.y = pos.y
        ps.point.z = pos.z
        pst = tf_listener.transformPoint("blender", ps)
        return pst.point


def pos_to_target(pos, speed):
    msg = Target()
    msg.x = max(0.3, pos.x)
    msg.y = pos.y if not math.isnan(pos.y) else 0
    msg.z = pos.z if not math.isnan(pos.z) else 0
    msg.z = max(-0.3, min(0.3, msg.z))
    if pos.x < 0.3:
        msg.z = 0
        msg.y = 0
    msg.speed = speed
    return msg


def SetGazeFocus(pos, speed, ts, frame_id='robot'):
    try:
        pos = get_blender_pos(pos, ts, frame_id)
        msg = pos_to_target(pos, speed)
        gaze_focus_pub.publish(msg)
    except Exception as e:
        print("Gaze focus exception: {}".format(e))

def SetHeadFocus(pos, speed, ts, frame_id='robot', proportional=True, head_yaw_movement=0.6, head_pitch_movement=0.6):
    try:
        pos = get_blender_pos(pos, ts, frame_id)
        msg = pos_to_target(pos, speed)
        if proportional:
            msg.y = msg.y * head_yaw_movement
            msg.z = msg.z * head_pitch_movement

        head_focus_pub.publish(msg)
    except Exception as e:
        print("Head focus exception: {}".format(e))


def UpdateGaze(pos, ts, frame_id="realsense"):
    SetGazeFocus(pos, 5.0, ts, frame_id)
    SetHeadFocus(pos, 1, ts, frame_id)


def calculate_normalized_px(u,v):
    x = 1.0
    y = (320-u)/320
    z = (240-v)/480
    pos = PosStruct()
    pos.set_xyz(x,y,z)
    return pos


rate = rospy.Rate(30)
while not rospy.is_shutdown():
    print('-----------')
    u = eval(input('u (px):'))
    v = eval(input('v (px):'))
    pos =calculate_normalized_px(u,v)
    UpdateGaze(pos, rospy.Time.now()-rospy.Time.from_sec(0.06),'realsense')
    rate.sleep()