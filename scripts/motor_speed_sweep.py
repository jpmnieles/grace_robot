import os
import sys
sys.path.append(os.getcwd())

import pickle
import numpy as np
from datetime import datetime

from scservo_sdk import *
from grace.driver import *
from grace.utils import *


def deg2intmotor(deg):
    int_motor = deg*(4096/360)
    return int_motor

def intmotor2deg(int_motor):
    deg = int_motor*(360/4096)
    return deg

def generate_sequence(max_amplitude):
    list1 = list(range(1,24))
    list2 = []
    for i in range(2,max_amplitude+1,2):
        list2.append(round(deg2intmotor(i)))
    int_motor_list = list1 + list2
    deg_motor_list = ["%.4f"%(intmotor2deg(x)) for x in int_motor_list]
    return int_motor_list, deg_motor_list


def save_pickle_data(data, filename: str):
    # Making Directory
    filepath = os.path.join(os.path.abspath(""), "results", filename)

    # Saving to Pickle File
    with open(filepath + ".pickle", 'wb') as file:
        pickle.dump(data, file)
    print('Data saved in:', filepath + ".pickle")
    return filepath + ".pickle"


def main(motor_id, max_amplitude, trials):

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    driver = Feetech(motor_id)
    if motor_id == 15:
        motor_max = 2190
    else:
        motor_max = driver._motor_limits['int_max']
    motor_name = driver.motor_name

    int_motor_list, deg_motor_list = generate_sequence(max_amplitude)

    title_str = dt_str + '_MotorSweepSpeed_' + motor_name + '_' + '%damp_'%(max_amplitude) + '%dtrials'%(trials) 
    pickle_data = {
        "title": title_str,
        "datetime": dt_str,
        "data_list": [],
        "data_label": deg_motor_list
    }


    # Reset
    start_ts, end_ts, elapsed_time = driver.set_goal_position(motor_max)
    time.sleep(0.75)

    for i,int_motor in enumerate(int_motor_list):
        int_position = motor_max - int_motor
        temp_dict = {
            "degree": deg_motor_list[i],
            "start_position": motor_max,
            "goal_position": int_position,
            "delta_position": int_motor,
            "tx_data": [],
            "rx_data": []
        }
        for trial in range(trials):
            rx_list = []
            start_ts, end_ts, elapsed_time = driver.set_goal_position(int_position)
            temp_dict["tx_data"].append({"timestamp": start_ts, 
                            "end_timestamp": end_ts,
                            "elapsed_time": elapsed_time})

            total_time = 0
            while(total_time <= 0.75):
                present_position, present_speed, ts = driver.get_present_position_speed()
                total_time = (datetime.fromtimestamp(ts) - datetime.fromtimestamp(start_ts)).total_seconds()
                rx_dict = {
                    "timestamp": ts,
                    "present_position": present_position,
                    "present_speed": present_speed
                }
                rx_list.append(rx_dict)
                print("[%f] Present position: %d, Present speed: %d" % (ts, present_position, present_speed))
            temp_dict["rx_data"].append(rx_list)

            # Reset
            start_ts, end_ts, elapsed_time = driver.set_goal_position(motor_max)
            time.sleep(0.75)

        pickle_data["data_list"].append(temp_dict)

    save_pickle_data(pickle_data, filename=title_str)    


if __name__ == "__main__":
    main(motor_id=15, max_amplitude=22, trials=5)