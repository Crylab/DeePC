import track
import model
import numpy as np

class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_sum = 0
        self.last_error = 0

    def update(self, error):
        # Proportional term
        p = self.kp * error

        # Integral term
        self.error_sum += error
        i = self.ki * self.error_sum

        # Derivative term
        d = self.kd * (error - self.last_error)
        self.last_error = error

        # Calculate the PID output
        output = p + i + d

        return output
    
    def reset(self):
        self.error_sum = 0
        self.last_error = 0
    
class PID_Tracking(track.Abstrack_tracking):
    def __init__(self, parameters: dict = {}) -> None:
        super().__init__(parameters)
        
        self.set_default_parameters(parameters, 'heading_loop', [1.0, 0.0, 0.0])
        self.set_default_parameters(parameters, 'velocity_loop', [1.0, 0.0, 0.0])
        self.set_default_parameters(parameters, 'direction_loop', [0.1, 0.0, 0.0])
        self.set_default_parameters(parameters, 'distance_loop', [0.1, 0.0, 0.0])
        
        self.algorithm = 'pid'
        
        self.direction_loop = PID(*self.parameters['direction_loop'])
        self.distance_loop = PID(*self.parameters['distance_loop'])
        self.velocity_loop = PID(*self.parameters['velocity_loop'])
        self.heading_loop = PID(*self.parameters['heading_loop'])
        
        
    def tracking_initialization(self) -> None:
        pass
    
    def tracking_termination(self) -> None:
        pass
    
    def control_step(self) -> model.Racecar_Action:
                
        me = self.past_states[-1]
        aim = self.reference_states[0]
        
        dx = aim.x - me.x
        dy = aim.y - me.y
        
        vector = np.array([dx, dy])
        rot_mat = np.array(
            [[np.cos(-me.heading), -np.sin(-me.heading)], 
            [np.sin(-me.heading), np.cos(-me.heading)]]
        )
        vector_rot = np.dot(rot_mat, vector)
        delta = np.arctan2(vector_rot[1], vector_rot[0])
        dist = np.sqrt(dx ** 2 + dy ** 2)
        
        action = model.Racecar_Action()
        
        sig1 = self.direction_loop.update(delta)
        
        sig2 = aim.heading - me.heading+sig1
        action.steering = self.heading_loop.update(sig2)

        sig3 = self.distance_loop.update(dist)
        sig4 = aim.speed - me.speed + sig3
        action.throttle = self.velocity_loop.update(sig4)
        
        return action
        
