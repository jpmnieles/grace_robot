import time
import rospy
from hr_msgs.msg import TargetPosture, MotorStateList


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
                    self._motor_state[i] = x
            
    def set_motor_names(self, names: list):
        self.names = names
        self.num_names = len(self.names)

    @property
    def state(self):
        return self._motor_state

    def move(self, values):
        args = {"names":self.names, "values":values}
        self.publisher.publish(TargetPosture(**args))

    def exit(self):
        rospy.signal_shutdown('End of Node')


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

