import rospy
from std_msgs.msg import String
import keyboard

def key_check_node():
    rospy.init_node('key_check_node')
    pub = rospy.Publisher('key_press', String, queue_size=10)

    rate = rospy.Rate(5)  # Adjust the rate according to your needs

    while not rospy.is_shutdown():
        if keyboard.is_pressed('`'):  # Replace 'key' with the desired key
            key_press = '`'  # Replace 'key' with the desired key
            rospy.loginfo("Key '{}' pressed".format(key_press))
            pub.publish(key_press)

        rate.sleep()

if __name__ == '__main__':
    try:
        key_check_node()
    except rospy.ROSInterruptException:
        pass