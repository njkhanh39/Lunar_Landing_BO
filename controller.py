import numpy as np

class PIDController:
    """
    PID controller for LunarLander with continuous action mapping
    """
    def __init__(self, params):
        # params = [kp_alt, kd_alt, kp_ang, kd_ang]
        self.kp_alt = params[0]
        self.kd_alt = params[1]
        self.kp_ang = params[2]  # This should be NEGATIVE for stability
        self.kd_ang = params[3]
        
        # Action thresholds (tunable)
        self.alt_threshold = 0.3
        self.ang_threshold = 0.15
        
    def get_action(self, state):
        """
        state: [x, y, vel_x, vel_y, angle, ang_vel, leg1, leg2]
        returns: int (0, 1, 2, 3)
        """
        y_pos = state[1]
        y_vel = state[3]
        angle = state[4]
        ang_vel = state[5]
        
        # Target descent velocity (negative for descending)
        target_y_vel = -0.5
        
        # Vertical control: PD for altitude
        alt_error = target_y_vel - y_vel
        alt_todo = self.kp_alt * alt_error + self.kd_alt * (0 - y_pos)
        
        # Angular control: PD for angle (keep upright at 0 radians)
        ang_error = 0 - angle  # Want angle = 0
        ang_todo = self.kp_ang * ang_error + self.kd_ang * (0 - ang_vel)
        
        # Action selection with hysteresis
        # Priority: orientation first, then altitude
        
        # If angle is significantly off, correct it
        if abs(ang_todo) > self.ang_threshold:
            if ang_todo > 0:
                return 1  # Fire left engine (rotate clockwise)
            else:
                return 3  # Fire right engine (rotate counter-clockwise)
        
        # If vertical control requires action
        if alt_todo > self.alt_threshold:
            return 2  # Fire main engine (upward thrust)
        
        # Default: do nothing
        return 0