import numpy as np

class PIDController:
    """
    controller for the rocket, has 4 params
    """
    def __init__(self, params):
        
        # params = [kp_alt, kd_alt, kp_ang, kd_ang]
        
        # altitude
        self.kp_alt = params[0]  # proportional
        self.kd_alt = params[1]  # derivative

        # angle
        self.kp_ang = params[2]  # proportional
        self.kd_ang = params[3]  # derivative
        
    def get_action(self, state):
        """
        state: [x, y, vel_x, vel_y, angle, ang_vel, leg1, leg2]
        returns: int (0, 1, 2, 3)
        """

        y_pos = state[1]
        y_vel = state[3]
        angle = state[4]
        ang_vel = state[5]


        # target: want y_pos to be 0 (ground)
        # formula: F = (kp * error) + (kd * velocity_error)
        # we need positive force to counteract gravity.
        

        target_y_vel = -0.5  # descent speed
        
        # vertical speed adjust
        alt_todo = (self.kp_alt * (target_y_vel - y_vel)) + (self.kd_alt * (0 - y_pos))
        
        
        # we want angle to be 0 (upright)
        ang_todo = (self.kp_ang * (0 - angle)) + (self.kd_ang * (0 - ang_vel))


        action = 0

        # priority 1: orientation 
        if ang_todo > 0.05:
            action = 1  
        elif ang_todo < -0.05:
            action = 3  
        
        # priority 2: altitude 
        else:
            if alt_todo > 0.05:
                action = 2  
        return action