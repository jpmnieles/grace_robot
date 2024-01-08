import rospy
from std_msgs.msg import String

import json
import time
import threading

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class GraceEnv(gym.Env):

    state = None

    def __init__(self, max_steps=10):
        super(GraceEnv, self).__init__()

        rospy.init_node('grace_state_subscriber')
        self.state_sub = rospy.Subscriber('/grace/state', String, self._grace_state_callback, queue_size=1)
        
        self.state_lock = threading.Lock()
        self.condition = threading.Condition()

        self.seed_value = 143
        # self.reset(0, 0)
        self.max_steps = max_steps  # Maximum number of steps before termination of episode
        self.x_min = -20  # Minimum x-coordinate boundary
        self.x_max = 20  # Maximum x-coordinate boundary
        self.step_count = 0  # Step count
        0.30679615757712825
        
        self.action_space = spaces.Box(low=np.array([-0.30679615757712825, -0.30679615757712825, -0.5368932757599744]), 
                                       high=np.array([0.30679615757712825, 0.30679615757712825, 0.38349519697141027]),
                                       shape=(3,), dtype=np.float32)  # Continuous action space of size 1
        self.observation_space = spaces.Box(low=-1, high=-1, shape=(3,), dtype=np.float32)

    def _grace_state_callback(self, msg):
        with self.condition:
            dict_data = json.loads(msg.data)
            with self.state_lock:
                self.state = dict_data
            # print(dict_data)
            self.condition.notify()

    def seed(self, seed_value):
        self.seed_value = seed_value
        np.random.seed(seed_value)

    def reset(self):
        self.step_count = 0  # Step count
        self.current_phi = self.m * self.current_eta
        self.target_phi = np.random.uniform(self.m*-13, self.m*13)
        self.delta_phi = self.target_phi - self.current_phi
        return np.array([self.current_theta, self.current_eta, self.delta_phi])

    def step(self, action):
        angle = action[0]

        # Apply boundary constraint for x-coordinate
        next_theta = self.current_theta + angle
        next_theta = np.clip(next_theta, self.x_min, self.x_max)
        next_eta = self.current_eta + max(0, next_theta-self.current_eta) - max(0, self.current_eta-self.backlash-next_theta)

        # Update the agent's position
        self.current_theta = next_theta
        self.current_eta = next_eta
        self.current_phi = self.m * self.current_eta
        self.step_count += 1  # Step count

        done = False
        if self.step_count >= self.max_steps or np.abs(self.target_phi - self.current_phi) < 0.2:
            done = True

        reward = -np.abs(self.target_phi - self.current_phi)  # Negative absolute difference between the current y and the target y
        self.delta_phi = self.target_phi - self.current_phi
        return np.array([self.current_theta, self.current_eta, self.delta_phi]), reward, done, {}


if __name__ == '__main__':

    # Test the environment
    env = GraceEnv()

    for i in range(5):
        with env.condition:
            env.condition.wait()
            with env.state_lock:
                print(env.state)