import os
import sys
import yaml
import math


def motor_int_to_angle(motor, position, degrees):
    if degrees:
        unit = 360
    else:
        unit = math.pi
    angle = ((position-motors_dict[motor]['init'])/4096)*unit
    return angle


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
