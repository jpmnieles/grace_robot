import os
import sys
import time
import yaml
import rospy
from hr_msgs.msg import TargetPosture, MotorStateList
from rospy_message_converter import message_converter


class ROSMotorClient(object):
    

    def __init__(self, motors=["EyeTurnLeft", "EyeTurnRight", "EyesUpDown"]):
        rospy.init_node('hr_motor_client')
        self.set_motor_names(motors)
        self._motor_state = [None]*self.num_names
        self.subscriber = rospy.Subscriber('/hr/actuators/motor_states', MotorStateList, self._capture_state)
        self.publisher = rospy.Publisher('/hr/actuators/pose', TargetPosture, queue_size=10)
        time.sleep(1)
        self.state

    def _capture_state(self, msg):
        """Callback for capturing motor state
        """
        self._msg = msg
        for x in msg.motor_states:
            for i, name in enumerate(self.names):
                if x.name == name:
                    self._motor_state[i] = message_converter.convert_ros_message_to_dictionary(x)

    def _capture_limits(self, motor):
        min = motors_dict[motor]['motor_min']
        init = motors_dict[motor]['init']
        max = motors_dict[motor]['motor_max']
        limits = {'min': min, 'init': init, 'max': max}
        return limits

    def set_motor_names(self, names: list):
        self.names = names
        self.num_names = len(self.names)
        self._motor_limits = {motor: self._capture_limits(motor) for motor in self.names}

    @property
    def state(self):
        return self._motor_state

    def move(self, values):
        args = {"names":self.names, "values":values}
        self.publisher.publish(TargetPosture(**args))

    def exit(self):
        rospy.signal_shutdown('End of Node')


# Loading Motors Yaml File
with open(os.path.join(os.getcwd(),'config', 'head','motors.yaml'), 'r') as stream:
    try:
        head_dict = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        print(error)

with open(os.path.join(os.getcwd(),'config', 'body','motors.yaml'), 'r') as stream:
    try:
        body_dict = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        print(error)

motors_dict = {}
motors_dict.update(head_dict['motors'])
motors_dict.update(body_dict['motors'])


if __name__ == '__main__':

    # Instantiation
    client = ROSMotorClient(["EyeTurnLeft", "EyeTurnRight", "EyesUpDown"])

    # State
    start = time.time()
    print(client.state)
    end = time.time()
    print("State Elapsed Time:", end-start)

    # Move
    values = eval(input("Enter the motor commands in list:"))
    start = time.time()
    client.move(values)
    end = time.time()
    print("Move Command Elapsed Time:", end-start)

    # State
    start = time.time()
    print(client.state)
    end = time.time()
    print("State Elapsed Time:", end-start)
