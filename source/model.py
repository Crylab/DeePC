
import numpy as np
from termcolor import colored
from abc import ABC, abstractmethod

class Racecar_Action:
    def __init__(
            self, 
            throttle: float = 0.0, 
            steering: float = 0.0, 
        ):
        self.throttle = throttle
        self.steering = steering
        
    def saturate(self, max_steering: float = 1.57):
        saturated_action = Racecar_Action()
        saturated_action.throttle = np.clip(self.throttle, -1.0, 1.0)
        saturated_action.steering = np.clip(self.steering, -max_steering, max_steering)
        return saturated_action

class Racecar_State:
    def __init__(
            self, 
            x: float = 0.0, 
            y: float = 0.0, 
            speed: float = 0.0, 
            heading: float = 0.0
        ):
        self.x = x
        self.y = y
        self.speed = speed
        self.heading = heading
    
    def position(self):
        return np.array([self.x, self.y])

class Racecar_State_3DOF(Racecar_State):
    
    def __init__(
            self, 
            x: float = 0.0,
            y: float = 0.0, 
            speed: float = 0.0, 
            heading: float = 0.0
        ):
        super().__init__(x, y, speed, heading)
        self.velocity_x = self.speed
        self.velocity_y = 0.0
        self.velocity_heading = 0.0
        
    def get_speed(self) -> float:
        return np.linalg.norm(self.velocity())
        
    def get_Racecar_State(self):
        return Racecar_State(
            self.x, 
            self.y, 
            self.get_speed(), 
            self.heading
            )
    
    def velocity(self):
        return np.array([self.velocity_x, self.velocity_y])

class Abstract_model(ABC):
    @abstractmethod
    def __init__(self, parameters: dict = {}) -> None:
        """
        Constructor of model class build object of the model
        
        Parameters:
            parameters (dict): Dictionary of model's parameters
            
        Returns:
            None        
        """        
        super().__init__()
    
    def Initialization(self, state: Racecar_State) -> None:
        """
        This function is define initial condition of the racecar
        
        Parameters:
            state (Racecar_State): Inital state of the racecar to setup
            
        Returns:
            None        
        """        
        self.state = state
    
    @abstractmethod
    def Step(self, action: Racecar_Action) -> Racecar_State:
        """
        This function is realizing one time step of racecar simulation
        
        Parameters:
            action (Racecar_Action): Racecar action to apply at current simulation step
            
        Returns:
            state (Racecar_State): Racecar state after simulation step        
        """     
        pass
    
class Kinematic_model(Abstract_model):
    def __init__(self, parameters: dict = {}) -> None:
        self.parameters = {}
        if 'wheelbase' in parameters.keys():
            self.parameters['wheelbase'] = parameters['wheelbase']
        else:
            self.parameters['wheelbase'] = 3.135  # m
        if 'engine_force' in parameters.keys():
            self.parameters['engine_force'] = parameters['engine_force']
        else:
            self.parameters['engine_force'] = 8800.0  # Newtons
        if 'mass' in parameters.keys():
            self.parameters['mass'] = parameters['mass']
        else:
            self.parameters['mass'] = 896.0  # kg
        if 'brake_force' in parameters.keys():
            self.parameters['brake_force'] = parameters['brake_force']
        else:
            self.parameters['brake_force'] = self.parameters['brake_force']
        if 'max_steering_angle' in parameters.keys():
            self.parameters['max_steering_angle'] = parameters['max_steering_angle']
        else:
            self.parameters['max_steering_angle'] = 0.26 # Radians
        if 'dt' in parameters.keys():
            self.parameters['dt'] = parameters['dt']
        else:
            self.parameters['dt'] = 0.01 # Seconds
            
        self.state = Racecar_State()        
        super().__init__()
        
    def Step(self, action: Racecar_Action) -> Racecar_Action:
        
        # Short named constants
        engine = self.parameters['engine_force']
        brake  = self.parameters['brake_force']
        mass   = self.parameters['mass']
        L      = self.parameters['wheelbase']
        dt     = self.parameters['dt']
        
        # Get actions
        act = action.saturate(self.parameters['max_steering_angle'])
        
        # Acceleration computation based on 
        # throttle and racecar capabilities
        acceleration = act.throttle * (engine if act.throttle > 0.0 else brake) / mass
        
        # Racecar speed update
        self.state.speed += acceleration * dt
        
        # Short name state variables
        speed = self.state.speed
        theta = self.state.heading
        
        self.state.x += speed * np.cos(theta) * dt
        self.state.y += speed * np.sin(theta) * dt
        self.state.heading += (speed / L) * np.tan(act.steering) * dt
        
        return self.state

class Dynamic_model(Kinematic_model):
    def __init__(self, parameters: dict = {}):
        super().__init__(parameters=parameters)
        
        # Tires parameters
        if 'pacejka_D' in parameters.keys():
            self.parameters['pacejka_D'] = parameters['pacejka_D']
        else:
            self.parameters['pacejka_D'] = 1.0
        if 'pacejka_C' in parameters.keys():
            self.parameters['pacejka_C'] = parameters['pacejka_C']
        else:
            self.parameters['pacejka_C'] = 1.1
        if 'pacejka_B' in parameters.keys():
            self.parameters['pacejka_B'] = parameters['pacejka_B']
        else:
            self.parameters['pacejka_B'] = 25.0
            
        # Aerodynamic parameters    
        if 'drag_coefficient' in parameters.keys():
            self.parameters['drag_coefficient'] = parameters['drag_coefficient']
        else:
            self.parameters['drag_coefficient'] = 1.35 # m^2
        if 'downforce_coefficient' in parameters.keys():
            self.parameters['downforce_coefficient'] = parameters['downforce_coefficient']
        else:
            self.parameters['downforce_coefficient'] = 4.31 # m^2
            
        # other physical parameters
        if 'inertia_moment' in parameters.keys():
            self.parameters['inertia_moment'] = parameters['inertia_moment']
        else:
            self.parameters['inertia_moment'] = 1500 # kg m^2
        if 'power' in parameters.keys():
            self.parameters['power'] = parameters['power']
        else:
            self.parameters['power'] = 462334 # W
                    
        self.state = Racecar_State_3DOF()
        
        # Constant: Free fall acceleration
        self.G = 9.81 # m/s^2
        # Constant: Air density
        self.RHO = 1.225 # kg/m^3
        
    def weight(self) -> float:
        speed = self.state.get_speed()
        
        # Short named constants
        ClA = self.parameters['downforce_coefficient']
        mass = self.parameters['mass']
        
        downforce = 0.5 * self.RHO * ClA * speed ** 2
        weight = mass * self.G
        
        return weight + downforce

    def Step(self, action: Racecar_Action) -> Racecar_Action:
        
        # Short named constants
        power = self.parameters['engine_force']
        brake  = self.parameters['brake_force']
        mass   = self.parameters['mass']
        L2     = self.parameters['wheelbase']/2
        dt     = self.parameters['dt']
        CdA    = self.parameters['drag_coefficient']
        inertia = self.parameters['inertia_moment']
        
        # Tire model constants
        pacD =  self.parameters['pacejka_D']
        pacC =  self.parameters['pacejka_C']
        pacB =  self.parameters['pacejka_B']
        
        # Get actions
        act = action.saturate(self.parameters['max_steering_angle'])
        
        # Short name state variables
        vel_x = self.state.velocity_x
        vel_y = self.state.velocity_y
        vel_h = self.state.velocity_heading        
           
        # Slipping angles
        alpha_R = np.arctan2((vel_y-(L2*vel_h)), vel_x)
        alpha_F = np.arctan2((vel_y+(L2*vel_h)), vel_x) - act.steering

        # Pacejika Lateral Forces computation
        F_R_y = - pacD * np.sin(pacC * np.arctan(pacB*alpha_R)) * self.weight()/2 # Rear wheel
        F_F_y = - pacD * np.sin(pacC * np.arctan(pacB*alpha_F)) * self.weight()/2 # Front wheel
        
        # Force generated by engine or brakes
        engine = power / (self.state.get_speed() if self.state.get_speed() < 1.0 else 1.0)
        F_x = act.throttle * (engine if act.throttle > 0.0 else brake * np.sign(self.state.velocity_x))

        # Maximum friction force available with current conditions   
        max_friction_force = pacD * self.weight() / 2
        F_x = np.clip(F_x, -max_friction_force, max_friction_force)
        
        # Drag forces computation
        F_x_aero = -0.5 * np.sign(vel_x) * CdA * self.RHO * vel_x**2
        F_y_aero = -0.5 * np.sign(vel_y) * CdA * self.RHO * vel_y**2

        # Acceleration
        a_x = ((F_x - (F_F_y*np.sin(act.steering)) + F_x_aero)/mass) + (vel_y * vel_h)
        a_y = ((F_R_y + (F_F_y*np.cos(act.steering)) + F_y_aero)/mass) - (vel_x * vel_h) 

        # Heading angular acceleration
        a_h = ((F_F_y * L2 * np.cos(act.steering)) - (F_R_y * L2)) / inertia

        # Full stop condition
        if abs(a_x * dt) > abs(vel_x) and act.throttle<0.0:
            self.state.velocity_x = 0.0
        else:
            self.state.velocity_x += a_x * dt
            
        self.state.velocity_y += a_y * dt
        
        self.state.velocity_heading += a_h * dt
        
        velocity = self.state.velocity()
        
        # Rotation matrix implementation
        cos_h = np.cos(self.state.heading)
        sin_h = np.sin(self.state.heading)
        rot_mat = np.array([cos_h, -sin_h],
                           [sin_h, cos_h])
        
        position_dot = np.matmul(rot_mat, velocity) * dt
        
        self.state.x += position_dot[0]
        self.state.y += position_dot[1]
        self.state.heading += self.state.velocity_heading * dt
        
        return self.state.get_Racecar_State()
        