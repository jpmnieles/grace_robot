"""Module for the Baseline Policy
"""

import os
import sys
sys.path.append(os.getcwd())

import math
import copy
import threading

from grace.utils import *


class BaselineCalibration(object):

    init_buffer = {
        't-1': {
            'cmd': {
                'EyeTurnLeft': None,
                'EyeTurnRight': None,
                'EyesUpDown': None
            },
            'state': {
                'EyeTurnLeft': None,
                'EyeTurnRight': None,
                'EyesUpDown': None
            },
            'hidden': {
                'EyeTurnLeft': None,
                'EyeTurnRight': None,
                'EyesUpDown': None
            },
        },
        't': {
            'cmd': {
                'EyeTurnLeft': None,
                'EyeTurnRight': None,
                'EyesUpDown': None
            },
            'state': {
                'EyeTurnLeft': None,
                'EyeTurnRight': None,
                'EyesUpDown': None
            },
            'hidden': {
                'EyeTurnLeft': None,
                'EyeTurnRight': None,
                'EyesUpDown': None
            },
        },
        't+1': {
            'cmd': {
                'EyeTurnLeft': None,
                'EyeTurnRight': None,
                'EyesUpDown': None
            }
        }
    }


    def __init__(self, lock) -> None:
        self.lock = lock
        self.camera_mtx = load_camera_mtx()
        self.calib_params = load_json('config/calib/calib_params.json')
        self.toggle_backlash(False)
        self.reset_buffer()

    def reset_buffer(self):
        with self.lock:
            self.buffer = self.init_buffer.copy()

    def store_latest_state(self, latest_state):
        self.buffer['t']['cmd'] = copy.deepcopy(self.buffer['t+1']['cmd'])
        self.buffer['t-1'] = copy.deepcopy(self.buffer['t'])

        self.buffer['t']['state']['EyeTurnLeft'] = latest_state[0]
        self.buffer['t']['state']['EyeTurnRight'] = latest_state[1]
        self.buffer['t']['state']['EyesUpDown'] = latest_state[2]

    def _px_to_deg_fx(self, x, eye:str):
        """eye (str): select from ['left_eye', 'right_eye']
        """
        theta = math.atan(x/self.camera_mtx[eye]['fx'])
        theta = math.degrees(theta)
        return theta

    def _px_to_deg_fy(self, y, eye:str):
        """eye (str): select from ['left_eye', 'right_eye']
        """
        theta = math.atan(y/self.camera_mtx[eye]['fy'])
        theta = math.degrees(theta)
        return theta

    def compute_left_hidden_state(self):
        theta_l_pan = self.buffer['t']['state']['EyeTurnLeft']['angle']
        backlash = self.calib_params['left_eye']['backlash']
        eta_tminus1 = self.buffer['t-1']['hidden']['EyeTurnLeft']
        if eta_tminus1 is None:
            eta_tminus1 = self.buffer['t']['state']['EyeTurnLeft']['angle']
            self.buffer['t-1']['hidden']['EyeTurnLeft'] = eta_tminus1
        eta_t = eta_tminus1 + max(0, theta_l_pan-eta_tminus1) - max(0, eta_tminus1-backlash-theta_l_pan)
        self.buffer['t']['hidden']['EyeTurnLeft'] = eta_t
        return eta_t
    
    def compute_right_hidden_state(self):
        theta_r_pan = self.buffer['t']['state']['EyeTurnRight']['angle']
        backlash = self.calib_params['right_eye']['backlash']
        eta_tminus1 = self.buffer['t-1']['hidden']['EyeTurnRight']
        if eta_tminus1 is None:
            eta_tminus1 = self.buffer['t']['state']['EyeTurnRight']['angle']
            self.buffer['t-1']['hidden']['EyeTurnRight'] = eta_tminus1
        eta_t = eta_tminus1 + max(0, theta_r_pan-eta_tminus1) - max(0, eta_tminus1-backlash-theta_r_pan)
        self.buffer['t']['hidden']['EyeTurnRight'] = eta_t
        return eta_t

    def compute_left_eye_cmd(self, dx, dy):
        theta_l_pan = self.buffer['t']['state']['EyeTurnLeft']['angle']
        slope_l_pan = self.calib_params['left_eye']['slope']
        theta_l_tilt = self.buffer['t']['state']['EyesUpDown']['angle']
        slope_l_tilt =  self.calib_params['tilt_eyes']['slope']

        delta_eta_l_pan = self._px_to_deg_fx(dx, 'left_eye')/slope_l_pan
        delta_eta_l_tilt = self._px_to_deg_fy(dy, 'left_eye')/slope_l_tilt

        if self.en_backlash:
            eta_l_pan = self.compute_left_hidden_state()
            backlash = self.calib_params['left_eye']['backlash']
            
            if delta_eta_l_pan > 1.3:
                theta_l_pan_tplus1 = eta_l_pan + delta_eta_l_pan
            elif delta_eta_l_pan < -1.3:
                theta_l_pan_tplus1 = eta_l_pan - backlash + delta_eta_l_pan
            elif delta_eta_l_pan == 0:
                theta_l_pan_tplus1 = theta_l_pan
            else:
                theta_l_pan_tplus1 = theta_l_pan + delta_eta_l_pan
        else:
            theta_l_pan_tplus1 = theta_l_pan + delta_eta_l_pan
        theta_l_tilt_tplus1 = theta_l_tilt + delta_eta_l_tilt
        return theta_l_pan_tplus1, theta_l_tilt_tplus1

    def compute_right_eye_cmd(self, dx, dy):
        theta_r_pan = self.buffer['t']['state']['EyeTurnRight']['angle']
        slope_r_pan = self.calib_params['right_eye']['slope']
        theta_r_tilt = self.buffer['t']['state']['EyesUpDown']['angle']
        slope_r_tilt =  self.calib_params['tilt_eyes']['slope']

        delta_eta_r_pan = self._px_to_deg_fx(dx, 'right_eye')/slope_r_pan
        delta_eta_r_tilt = self._px_to_deg_fy(dy, 'right_eye')/slope_r_tilt

        if self.en_backlash:
            eta_r_pan = self.compute_right_hidden_state()
            backlash = self.calib_params['right_eye']['backlash']
            
            if delta_eta_r_pan > 1.3:
                theta_r_pan_tplus1 = eta_r_pan + delta_eta_r_pan
            elif delta_eta_r_pan < -1.3:
                theta_r_pan_tplus1 = eta_r_pan - backlash + delta_eta_r_pan
            elif delta_eta_r_pan == 0:
                theta_r_pan_tplus1 = theta_r_pan
            else:
                theta_r_pan_tplus1 = theta_r_pan + delta_eta_r_pan
        else:
            theta_r_pan_tplus1 = theta_r_pan + delta_eta_r_pan
        theta_r_tilt_tplus1 = theta_r_tilt + delta_eta_r_tilt
        return theta_r_pan_tplus1, theta_r_tilt_tplus1

    def toggle_backlash(self, en_backlash):
        self.en_backlash = en_backlash

    def compute_tilt_cmd(self, theta_l_tilt, theta_r_tilt, alpha_tilt=0.5):
        """alpha_tilt: percentage of theta right tilt
        """
        if theta_l_tilt is None:
            theta_tilt = theta_r_tilt
        elif theta_r_tilt is None:
            theta_tilt = theta_l_tilt
        elif theta_l_tilt is None and theta_r_tilt is None:
            theta_tilt = None
        else:
            theta_tilt = (1-alpha_tilt)*theta_l_tilt + alpha_tilt*theta_r_tilt
        return theta_tilt
    
    def store_cmd(self, theta_l_pan, theta_r_pan, theta_tilt):
        if theta_l_pan is None:
            theta_l_pan = self.buffer['t']['state']['EyeTurnLeft']['angle']
        if theta_r_pan is None:
            theta_r_pan = self.buffer['t']['state']['EyeTurnRight']['angle']
        if theta_tilt is None:
            theta_tilt = self.buffer['t']['state']['EyesUpDown']['angle']

        self.buffer['t+1']['cmd']['EyeTurnLeft'] = theta_l_pan
        self.buffer['t+1']['cmd']['EyeTurnRight'] = theta_r_pan
        self.buffer['t+1']['cmd']['EyesUpDown'] = theta_tilt