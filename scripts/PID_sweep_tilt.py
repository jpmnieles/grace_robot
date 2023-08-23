import os
import sys
sys.path.append(os.getcwd())

import pickle
import numpy as np
from datetime import datetime
import itertools

import matplotlib
matplotlib.use('TkAgg')


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scservo_sdk import *
from grace.driver import *
from grace.utils import *


## Helper Functions

def px_to_deg_fx(x):
    x = math.atan(x/578.75956519)  #RIGHT: fx = 578.75956519. fy = 578.60434183
    x = math.degrees(x)
    return x

def px_to_deg_fy(x):
    x = math.atan(x/578.60434183)  #RIGHT: fx = 578.75956519. fy = 578.60434183
    x = math.degrees(x)
    return x

def datalabel2num(pickle_dict, data_label):
    for i,x in enumerate(pickle_dict['data_label']):
        if x == data_label:
            return i

def deg2intmotor(deg):
    int_motor = deg*(4096/360)
    return int_motor

def intmotor2deg(int_motor):
    deg = int_motor*(360/4096)
    return deg

def generate_sequence(max_amplitude):
    int_list = [max_amplitude, 30, 15, 10, 5, 2, 1, 0.5, 0.2]
    int_motor_list = [round(deg2intmotor(x)) for x in int_list]
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

def random_colors(n):
    return np.random.rand(n,3)


## Main Function
def main(motor_id, max_amplitude):

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    driver = Feetech(motor_id)
    if motor_id == 14:
        motor_max = 2111
    elif motor_id == 15:
        motor_max = 2190
    elif motor_id == 16:
        motor_max = 2661
    else:
        motor_max = driver._motor_limits['int_max']
    motor_name = driver.motor_name


    # Sequence Generation
    indices = list(range(0,20,1))

    p_list = list(range(30,101,10))
    i_list = [0]
    d_list = [0]
    combinations = itertools.product(p_list,i_list,d_list)     

    int_motor_list, deg_motor_list = generate_sequence(max_amplitude)
    int_motor = int_motor_list[0]

    int_position_list = []
    for int_motor in int_motor_list:
        int_position_list.append(motor_max - int_motor)

    title_str = dt_str + '_PIDSweep_' + motor_name + '_' + '%damp_'%(max_amplitude)
    pickle_data = {
        "title": title_str,
        "datetime": dt_str,
        "data_list": [],
        "data_label": None
    }

    # Reset
    start_ts, end_ts, elapsed_time = driver.set_goal_position(motor_max)
    time.sleep(0.75)

    rand_colors = random_colors(len(indices))

    print(combinations)
    
    combination_list = []
    plt.figure()
    
    for ctr in range(len(indices)):
        plt.scatter(0,0, color=rand_colors[ctr], marker='.')

    ## Processing
    cnt = 0
    for combination in combinations:
        combination_list.append(str(combination))
        temp_dict = {
            "PID": str(combination),
            "start_position": motor_max,
            "goal_position": None,
            "delta_position": None,
            "tx_data": [],
            "rx_data": []
        }

        # Reset PID Values
        int_p_gain = 80
        int_i_gain = 10
        int_d_gain = 140
        _,_,_ = driver.set_p_gain(int_p_gain)
        _,_,_ = driver.set_i_gain(int_i_gain)
        _,_,_ = driver.set_d_gain(int_d_gain)

        # Reset
        _, _, _ = driver.set_goal_position(motor_max)
        time.sleep(0.75)


        for trial in range(len(int_motor_list)):

            # Setting of PID
            driver.set_p_gain(combination[0])
            driver.set_i_gain(combination[1])
            driver.set_d_gain(combination[2])

            # Reading of PID
            p_gain,p_ts = driver.get_p_gain()
            i_gain,i_ts = driver.get_i_gain()
            d_gain,d_ts = driver.get_d_gain()
            print('==',combination,'==')
            print("[%.6f] P Gain: %i" % (p_ts, p_gain))
            print("[%.6f] I Gain: %i" % (i_ts, i_gain))
            print("[%.6f] D Gain: %i" % (d_ts, d_gain))


            rx_list = []
            int_motor = int_motor_list[trial]
            int_position = motor_max - int_motor
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

            # Reset PID Values
            int_p_gain = 80
            int_i_gain = 1
            int_d_gain = 140
            _,_,_ = driver.set_p_gain(int_p_gain)
            _,_,_ = driver.set_i_gain(int_i_gain)
            _,_,_ = driver.set_d_gain(int_d_gain)

            # Reset
            start_ts, end_ts, elapsed_time = driver.set_goal_position(motor_max)
            time.sleep(0.75)
        
        pickle_data["data_list"].append(temp_dict)
        cnt += 1


    ## Saving
    pickle_data['data_label'] = combination_list
    save_pickle_data(pickle_data, filename=title_str)

if __name__ == "__main__":
    main(motor_id=16, max_amplitude=60)