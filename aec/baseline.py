"""Module for the Baseline Policy
"""


import math


class PanBacklash(object):


    fx = 569.4456315
    m = 1.6328
    B = 1.5
    sigma = (0.1009, 0.8439, 1.057)


    def __init__(self, m=None, B=None, sigma:tuple=None, phi=None) -> None:
        if m is not None:
            self.m = m  # Slope
        if B is not None:
            self.B = B  # Backlash value
        if sigma is not None:
            self.sigma = sigma  # Overshoot parameters

        self.theta = 0   # Motor position
        if phi is None:
            self.phi = 0.4  # Eye position  # TODO: Check if 0.4 is okay
        else:
            if self._check_validity(self.theta, phi):
                self.phi = phi


    def _check_validity(self, theta, phi):
        if (theta >= phi/self.m) or (theta <= (phi-self.B)/self.m):
            raise ValueError("Theta must be inside the range of Phi and (Phi-B)")
        else:
            return True


    def calc_overshoot(self, x):
        if x != 0:
            o = min(self.sigma[0]*abs(x)+self.sigma[1], self.sigma[2])/self.m
        else:
            o = 0
        return o
    

    def calc_cmd(self, delta_x, theta=None, phi=None):
        if phi is not None:
            self.phi = phi
        # self._check_validity(theta, self.phi)
        
        # Calculations
        xi = self.phi/self.m
        delta_xi = math.degrees(math.atan(delta_x/self.fx))/self.m
        tmp = xi + delta_xi
        o = self.calc_overshoot(x=delta_xi)
        
        # Piecewise linear function
        if delta_xi == 0:
            theta_new = theta
        elif delta_xi > 0:
            theta_new = tmp - o
        elif delta_xi < 0:
            theta_new = tmp + o - self.B
            
        # Forward Model Update
        delta_phi = self.m*(max(0, theta_new-xi+o) - max(0, xi-theta_new-self.B+o))
        phi_new = self.phi + delta_phi

        # Variable Update
        self.theta = theta_new
        self.phi = phi_new
        
        return theta_new, phi_new

    @property
    def cmd(self):
        return self.theta
    

    @property
    def position(self):
        return self.phi


class TiltPolicy(object):


    fy = 571.54490033
    m = 0.3910


    def __init__(self, m=None) -> None:
        if m is not None:
            self.m = m  # Slope
        self.theta = 0   # Motor position
        self.phi = 0  # Eye Position


    def calc_cmd(self, delta_y, theta=None):      
        # Calculations
        if theta is None:
            xi = self.phi/self.m
        else:
            xi = theta
            self.phi = self.m * theta
        delta_xi = math.degrees(math.atan(delta_y/self.fy))/self.m
        theta_new = xi + delta_xi
        phi_new = self.m*theta_new

        # Variable Update
        self.theta = theta_new
        self.phi = phi_new
        
        return theta_new, phi_new

    @property
    def cmd(self):
        return self.theta
    

    @property
    def position(self):
        return self.phi


if __name__ == "__main__":
    # Constants
    m = 1.21  # slope
    o_1 = 0.8816  # positive swing overshoot
    o_2 = 0.6706  # negative swing overshoot
    o = 0.7761
    b_1 = 0.4551  # right backlash
    b_2 = 4.9257  # left backlash
    B = 5.3808


    pan_backlash = PanBacklash(m=m, B=B, sigma=(o,o))

    pan_backlash.calc_cmd(delta_x=3.5480, theta=-7.2949, phi=-2.9900)
    print(pan_backlash.cmd)
    print(pan_backlash.position)