import rospy
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import JointState, Image

import json
import time
import math
import random
import threading

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from grace.rl_gaze import VisuoMotorNode


class GraceEnv(gym.Env):

    def __init__(self, max_steps=10):
        super(GraceEnv, self).__init__()

        rospy.init_node('grace_env')
        self.grace = VisuoMotorNode()
        self.seed_value = 143
        # self.reset(0, 0)
        self.max_steps = max_steps  # Maximum number of steps before termination of episode
        self.step_count = 0  # Step count
        
        self.action_space = spaces.Box(low=np.array([-0.30679615757712825, -0.30679615757712825, -0.5368932757599744]), 
                                       high=np.array([0.30679615757712825, 0.30679615757712825, 0.38349519697141027]),
                                       shape=(3,))  # Continuous action space of size 1
        self.observation_space = spaces.Box(low=-1, high=-1, shape=(3,))

    def seed(self, seed_value):
        self.seed_value = seed_value
        np.random.seed(seed_value)

    def reset(self, chess_idx=7):
        rospy.loginfo('Reset')
        self.step_count = 0  # Step count
        self.grace.set_chess_idx(chess_idx)
        rospy.sleep(1)
        self.grace.set_action(None)
        rospy.sleep(1)

        with self.grace.buffer_lock:
            tminus_state = self.grace.rl_state
            print(self.grace.rl_state)
        t_state = tminus_state
        state = np.array([
            math.radians(tminus_state['theta_left_pan']),
            math.radians(tminus_state['theta_right_pan']),
            math.radians(tminus_state['theta_tilt']),
            tminus_state['chest_cam_px'][0]/848,
            tminus_state['chest_cam_px'][1]/480,
            tminus_state['left_eye_px'][0]/640,
            tminus_state['left_eye_px'][1]/480,
            tminus_state['right_eye_px'][0]/640,
            tminus_state['right_eye_px'][1]/480,
            tminus_state['plan_phi_left_pan'],
            tminus_state['plan_phi_right_pan'],
            tminus_state['plan_phi_tilt'],
            math.radians(t_state['theta_left_pan']),
            math.radians(t_state['theta_right_pan']),
            math.radians(t_state['theta_tilt']),
            t_state['chest_cam_px'][0]/848,
            t_state['chest_cam_px'][1]/480,
            t_state['left_eye_px'][0]/640,
            t_state['left_eye_px'][1]/480,
            t_state['right_eye_px'][0]/640,
            t_state['right_eye_px'][1]/480,
            math.radians(t_state['plan_phi_left_pan']),
            math.radians(t_state['plan_phi_right_pan']),
            math.radians(t_state['plan_phi_tilt'])
        ])
        return state

    def step(self, action=[0,0,0]):
        self.step_count += 1  # Step count
        
        print('=============')
        with self.grace.buffer_lock:
            tminus_state = self.grace.rl_state
        t_state = tminus_state
        # action = [random.randint(-8,8), random.randint(-8,8), random.randint(-8,8)]
        print('Action:', action)


        self.grace.set_action(action)
        rospy.sleep(1)

        with self.grace.buffer_lock:
            t_state = self.grace.rl_state

        state = np.array([
            math.radians(tminus_state['theta_left_pan']),
            math.radians(tminus_state['theta_right_pan']),
            math.radians(tminus_state['theta_tilt']),
            tminus_state['chest_cam_px'][0]/848,
            tminus_state['chest_cam_px'][1]/480,
            tminus_state['left_eye_px'][0]/640,
            tminus_state['left_eye_px'][1]/480,
            tminus_state['right_eye_px'][0]/640,
            tminus_state['right_eye_px'][1]/480,
            tminus_state['plan_phi_left_pan'],
            tminus_state['plan_phi_right_pan'],
            tminus_state['plan_phi_tilt'],
            math.radians(t_state['theta_left_pan']),
            math.radians(t_state['theta_right_pan']),
            math.radians(t_state['theta_tilt']),
            t_state['chest_cam_px'][0]/848,
            t_state['chest_cam_px'][1]/480,
            t_state['left_eye_px'][0]/640,
            t_state['left_eye_px'][1]/480,
            t_state['right_eye_px'][0]/640,
            t_state['right_eye_px'][1]/480,
            math.radians(t_state['plan_phi_left_pan']),
            math.radians(t_state['plan_phi_right_pan']),
            math.radians(t_state['plan_phi_tilt'])
        ])

        reward = -(np.abs(t_state['left_eye_px'][0] - self.grace.calib_params['left_eye']['x_center'])
                   + np.abs(t_state['left_eye_px'][1] - self.grace.calib_params['left_eye']['y_center'])
                   + np.abs(t_state['right_eye_px'][0] - self.grace.calib_params['right_eye']['x_center'])
                   + np.abs(t_state['right_eye_px'][1] - self.grace.calib_params['right_eye']['y_center']))/4

        done = False
        if self.step_count >= self.max_steps or reward > -0.2:
            done = True

        return state, reward, done, {}


if __name__ == '__main__':

    # Test the environment
    env = GraceEnv()
    env.reset(0)

    for i in range(10):
        action = [random.randint(-8,8), random.randint(-8,8), random.randint(-31,0)]
        vals = env.step(action)
        print(vals)