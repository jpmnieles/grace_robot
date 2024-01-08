import rospy
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import JointState, Image

import json
import time
import random
import threading

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from grace.rl_gaze import VisuoMotorNode


class GraceEnv(gym.Env):

    state = None

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

    def reset(self):
        self.step_count = 0  # Step count
        self.current_phi = self.m * self.current_eta
        self.target_phi = np.random.uniform(self.m*-13, self.m*13)
        self.delta_phi = self.target_phi - self.current_phi
        return np.array([self.current_theta, self.current_eta, self.delta_phi])

    def step(self, action=0):
        self.step_count += 1  # Step count
        # reward = -np.abs(self.target_phi - self.current_phi)  # Negative absolute difference between the current y and the target y
        reward = 0

        print('=============')
        with self.grace.buffer_lock:
            print('PREV_STATE:', self.grace.rl_state)
        action = [random.randint(-8,8), random.randint(-8,8), random.randint(-8,8)]
        print('Action:', action)

        self.grace.set_action(action)
        print('set_action')
        rospy.sleep(1)

        with self.grace.buffer_lock:
            print('NEXT_STATE:', self.grace.rl_state)
        print('=============')

        done = False
        if self.step_count >= self.max_steps or reward > -0.2:
            done = True

        return np.array([0,0,0]), reward, done, {}


if __name__ == '__main__':

    # Test the environment
    env = GraceEnv()
    while not rospy.is_shutdown():
        env.step()
        rospy.sleep(0.5)