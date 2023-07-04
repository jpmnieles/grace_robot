import numpy as np
from datetime import datetime
import sys, tty, termios

from scservo_sdk import *
from grace.utils import *


class Feetech(object):


    REG_ADDR = {
        "TORQUE_ENABLE": 40,
        "GOAL_ACC": 41,
        "GOAL_POSITION": 42,
        "GOAL_SPEED": 46,
        "PRESENT_POSITION": 56,
    }


    def __init__(self, motor_id, baud_rate=1000000, device_name='/dev/ttyUSB0', protocol_end=0) -> None:
        # Initialize Variables
        self.motor_id = motor_id
        self.baudrate = baud_rate  # SCServo default baudrate : 1000000
        self.device_name  = device_name  # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"
        self.protocol_end = protocol_end  # SCServo bit end(STS/SMS=0, SCS=1)
        self.moving_status_threshold = 20

        # Instantiate Port and Packet Handler
        self.PortHandler = PortHandler(device_name)
        self.PacketHandler = PacketHandler(protocol_end)
    
        # Open port
        if self.PortHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            print("Press any key to terminate...")
            self._getch()
            quit()

        # Set port baudrate
        if self.PortHandler.setBaudRate(baud_rate):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            print("Press any key to terminate...")
            self._getch()
            quit()

        # Set Acceleration and Speed
        self.set_acceleration(int_acc=0)
        self.set_speed(int_speed=0)

        # Read limits
        self._motor_limits = self._capture_limits(motor_id)

    def _capture_limits(self, motor_id):
        motor = self._capture_motor_name(motor_id)
        self.motor_name = motor

        int_min = motors_dict[motor]['motor_min']
        int_init = motors_dict[motor]['init']
        int_max = motors_dict[motor]['motor_max']
        limits = {'int_min': int_min, 
                  'int_init': int_init, 
                  'int_max': int_max}
        return limits
    
    def _capture_motor_name(self, motor_id):
        for name in motors_dict.keys():
            if motors_dict[name]["motor_id"] == motor_id:
                return name

    def _getch(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def set_speed(self, int_speed):
        start_ts = datetime.timestamp(datetime.now())
        scs_comm_result, scs_error = self.PacketHandler.write2ByteTxRx(
            self.PortHandler, self.motor_id, self.REG_ADDR["GOAL_SPEED"], int_speed)
        if scs_comm_result != COMM_SUCCESS:
            print("%s" % self.PacketHandler.getTxRxResult(scs_comm_result))
        elif scs_error != 0:
            print("%s" % self.PacketHandler.getRxPacketError(scs_error))
        end_ts = datetime.timestamp(datetime.now())
        elapsed_time = (datetime.fromtimestamp(end_ts) - datetime.fromtimestamp(start_ts)).total_seconds()
        return start_ts, end_ts, elapsed_time

    def set_acceleration(self, int_acc):
        start_ts = datetime.timestamp(datetime.now())
        scs_comm_result, scs_error = self.PacketHandler.write2ByteTxRx(
            self.PortHandler, self.motor_id, self.REG_ADDR["GOAL_ACC"], int_acc)
        if scs_comm_result != COMM_SUCCESS:
            print("%s" % self.PacketHandler.getTxRxResult(scs_comm_result))
        elif scs_error != 0:
            print("%s" % self.PacketHandler.getRxPacketError(scs_error))
        end_ts = datetime.timestamp(datetime.now())
        elapsed_time = (datetime.fromtimestamp(end_ts) - datetime.fromtimestamp(start_ts)).total_seconds()
        return start_ts, end_ts, elapsed_time
        
    def set_goal_position(self, int_position):
        start_ts = datetime.timestamp(datetime.now())
        scs_comm_result, scs_error = self.PacketHandler.write2ByteTxRx(
            self.PortHandler, self.motor_id, self.REG_ADDR["GOAL_POSITION"], int_position)
        if scs_comm_result != COMM_SUCCESS:
            print("%s" % self.PacketHandler.getTxRxResult(scs_comm_result))
        elif scs_error != 0:
            print("%s" % self.PacketHandler.getRxPacketError(scs_error))
        end_ts = datetime.timestamp(datetime.now())
        elapsed_time = (datetime.fromtimestamp(end_ts) - datetime.fromtimestamp(start_ts)).total_seconds()
        return start_ts, end_ts, elapsed_time
    
    def get_present_position_speed(self):
        scs_present_position_speed, scs_comm_result, scs_error = self.PacketHandler.read4ByteTxRx(
            self.PortHandler, self.motor_id, self.REG_ADDR["PRESENT_POSITION"])
        if scs_comm_result != COMM_SUCCESS:
            print(self.PacketHandler.getTxRxResult(scs_comm_result))
        elif scs_error != 0:
            print(self.PacketHandler.getRxPacketError(scs_error))

        ts = datetime.timestamp(datetime.now())
        present_position = SCS_LOWORD(scs_present_position_speed)
        present_speed = SCS_HIWORD(scs_present_position_speed)
        return present_position, present_speed, ts


if __name__ == "__main__":
    left_pan = Feetech(motor_id=15)
    
    while(1):
        input_str = input("Enter Position: ")
        if input_str == 'q':  # Press Q to exit
            break
        else:
            int_position = eval(input_str)
        start_ts, end_ts, elapsed_time = left_pan.set_goal_position(int_position)
        print("[%f] Set goal position: %d, Elapsed time: %f" % (start_ts, int_position, elapsed_time))

        total_time = 0
        start_ts = datetime.timestamp(datetime.now())
        while(total_time <= 0.75):
            present_position, present_speed, ts = left_pan.get_present_position_speed()
            print("[%f] Present position: %d, Present speed: %d" % (ts, present_position, present_speed))
            total_time = (datetime.fromtimestamp(ts) - datetime.fromtimestamp(start_ts)).total_seconds()
            
        
