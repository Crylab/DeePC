import numpy as np
from abc import ABC, abstractmethod
import copy


class Racecar_Action:
    """
    This class defines the action for a racecar, including throttle and steering.

    Attributes:
        throttle (float): Throttle value ranging from -1.0 to 1.0.
        steering (float): Steering value ranging from -max_steering to max_steering.
    """

    def __init__(self, throttle: float = 0.0, steering: float = 0.0):
        """
        Initialize Racecar_Action with default values.

        Parameters:
            throttle (float): Initial throttle value. Default is 0.0.
            steering (float): Initial steering value. Default is 0.0.
        """
        self.throttle = throttle
        self.steering = steering

    def saturate(self, max_steering: float = 1.57):
        """
        Saturate the throttle and steering values within their respective limits.

        Parameters:
            max_steering (float): Maximum allowable steering value. Default is 1.57 radians.

        Returns:
            Racecar_Action: A new Racecar_Action instance with saturated values.
        """
        saturated_action = copy.copy(Racecar_Action())
        saturated_action.throttle = np.clip(self.throttle, -1.0, 1.0)
        saturated_action.steering = np.clip(self.steering, -max_steering, max_steering)
        return saturated_action

    def __str__(self):
        return "Throttle = {:.2f}".format(self.throttle) + "; Steering = {:.2f}".format(self.steering)+";"

class Racecar_State:
    """
    This class defines the state of a racecar, including position, speed, and heading.

    Attributes:
        x (float): X-coordinate position of the racecar.
        y (float): Y-coordinate position of the racecar.
        speed (float): Speed of the racecar.
        heading (float): Heading angle of the racecar.
    """

    def __init__(
        self, x: float = 0.0, y: float = 0.0, speed: float = 0.0, heading: float = 0.0
    ):
        """
        Initialize Racecar_State with default values.

        Parameters:
            x (float): Initial x-coordinate. Default is 0.0.
            y (float): Initial y-coordinate. Default is 0.0.
            speed (float): Initial speed. Default is 0.0.
            heading (float): Initial heading angle. Default is 0.0.
        """
        self.x = x
        self.y = y
        self.speed = speed
        self.heading = heading

    def position(self):
        """
        Get the position of the racecar as a numpy array.

        Returns:
            np.array: The position of the racecar [x, y].
        """
        return np.array([self.x, self.y])
    
    def __str__(self):
        return "x = {:.2f}".format(self.x) + "; y = {:.2f}".format(self.y) + "; Speed = {:.2f}".format(self.speed) + "; Heading = {:.2f}".format(self.heading)+";"

    def get_numpy(self):
        return np.array([self.x, self.y, self.speed, self.heading])

class Racecar_State_3DOF(Racecar_State):
    """
    This class extends Racecar_State to include 
    velocity components for a 3 degrees-of-freedom (DOF) model.

    Attributes:
        velocity_x (float): Velocity component in the x-direction.
        velocity_y (float): Velocity component in the y-direction.
        velocity_heading (float): Angular velocity of the heading.
    """

    def __init__(
        self, x: float = 0.0, y: float = 0.0, speed: float = 0.0, heading: float = 0.0
    ):
        """
        Initialize Racecar_State_3DOF with default values.

        Parameters:
            x (float): Initial x-coordinate. Default is 0.0.
            y (float): Initial y-coordinate. Default is 0.0.
            speed (float): Initial speed. Default is 0.0.
            heading (float): Initial heading angle. Default is 0.0.
        """
        super().__init__(x, y, speed, heading)
        self.velocity_x = self.speed
        del self.speed
        self.velocity_y = 0.0
        self.velocity_heading = 0.0
        
    def __init__(self, obj: Racecar_State = Racecar_State()):
        """
        Initialize Racecar_State_3DOF with default values.

        Parameters:
            x (float): Initial x-coordinate. Default is 0.0.
            y (float): Initial y-coordinate. Default is 0.0.
            speed (float): Initial speed. Default is 0.0.
            heading (float): Initial heading angle. Default is 0.0.
        """
        super().__init__(obj.x, obj.y, obj.speed, obj.heading)
        self.velocity_x = self.speed
        self.velocity_y = 0.0
        self.velocity_heading = 0.0

    def get_speed(self) -> float:
        """
        Calculate the current speed of the racecar.

        Returns:
            float: The current speed.
        """
        return np.linalg.norm(self.velocity())

    def get_Racecar_State(self):
        """
        Get the Racecar_State object corresponding to the current state.

        Returns:
            Racecar_State: The current state of the racecar.
        """
        return copy.copy(Racecar_State(self.x, self.y, self.get_speed(), self.heading))

    def velocity(self):
        """
        Get the velocity components as a numpy array.

        Returns:
            np.array: The velocity components [velocity_x, velocity_y].
        """
        return np.array([self.velocity_x, self.velocity_y])


class Abstract_model(ABC):
    """
    Abstract base class for racecar models.

    Methods:
        Initialization: Sets up the initial state of the racecar.
        Step: Abstract method for advancing the simulation by one time step.
    """

    @abstractmethod
    def __init__(self, parameters: dict = {}) -> None:
        """
        Constructor of model class to build the object of the model.

        Parameters:
            parameters (dict): Dictionary of model's parameters.

        Returns:
            None
        """
        super().__init__()

    def Initialization(self, state_in: Racecar_State = Racecar_State()) -> None:
        """
        Define the initial condition of the racecar.

        Parameters:
            state (Racecar_State): Initial state of the racecar to set up.

        Returns:
            None
        """
        self.state = copy.copy(state_in)

    @abstractmethod
    def Step(self, action: Racecar_Action) -> Racecar_State:
        """
        Realize one time step of racecar simulation.

        Parameters:
            action (Racecar_Action): Racecar action to apply at the current simulation step.

        Returns:
            state (Racecar_State): Racecar state after the simulation step.
        """
        pass


class Kinematic_model(Abstract_model):
    """
    Kinematic model for simulating the racecar's motion.

    Attributes:
        parameters (dict): Dictionary of model parameters.
        state (Racecar_State): Current state of the racecar.
    """

    def __init__(self, parameters: dict = {}) -> None:
        """
        Initialize the Kinematic_model with default or provided parameters.

        Parameters:
            parameters (dict): Dictionary of model parameters. Default values are used if not provided.

        Returns:
            None
        """
        self.parameters = {}
        if "wheelbase" in parameters.keys():
            self.parameters["wheelbase"] = parameters["wheelbase"]
        else:
            self.parameters["wheelbase"] = 3.135  # m
        if "engine_force" in parameters.keys():
            self.parameters["engine_force"] = parameters["engine_force"]
        else:
            self.parameters["engine_force"] = 8800.0  # Newtons
        if "mass" in parameters.keys():
            self.parameters["mass"] = parameters["mass"]
        else:
            self.parameters["mass"] = 896.0  # kg
        if "brake_force" in parameters.keys():
            self.parameters["brake_force"] = parameters["brake_force"]
        else:
            self.parameters["brake_force"] = 30764.0
        if "max_steering_angle" in parameters.keys():
            self.parameters["max_steering_angle"] = parameters["max_steering_angle"]
        else:
            self.parameters["max_steering_angle"] = 0.26  # Radians
        if "dt" in parameters.keys():
            self.parameters["dt"] = parameters["dt"]
        else:
            self.parameters["dt"] = 0.01  # Seconds

        self.state = Racecar_State()
        super().__init__()

    def Step(self, action: Racecar_Action) -> Racecar_State:
        """
        Simulate one time step of the kinematic model.

        Parameters:
            action (Racecar_Action): The action to apply (throttle and steering).

        Returns:
            Racecar_State: The updated state of the racecar after the simulation step.
        """
        # Short named constants
        engine = self.parameters["engine_force"]
        brake = self.parameters["brake_force"]
        mass = self.parameters["mass"]
        L = self.parameters["wheelbase"]
        dt = self.parameters["dt"]

        # Get actions
        act = action.saturate(self.parameters["max_steering_angle"])

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
    """
    Dynamic model extending the Kinematic_model to include more detailed physical effects.

    Attributes:
        parameters (dict): Dictionary of model parameters.
        state (Racecar_State_3DOF): Current state of the racecar with 3DOF.
    """

    def __init__(self, parameters: dict = {}):
        """


        Initialize the Dynamic_model with default or provided parameters.

        Parameters:
            parameters (dict): Dictionary of model parameters. Default values are used if not provided.

        Returns:
            None
        """
        super().__init__(parameters=parameters)

        # Tires parameters
        if "pacejka_D" in parameters.keys():
            self.parameters["pacejka_D"] = parameters["pacejka_D"]
        else:
            self.parameters["pacejka_D"] = 1.0
        if "pacejka_C" in parameters.keys():
            self.parameters["pacejka_C"] = parameters["pacejka_C"]
        else:
            self.parameters["pacejka_C"] = 1.1
        if "pacejka_B" in parameters.keys():
            self.parameters["pacejka_B"] = parameters["pacejka_B"]
        else:
            self.parameters["pacejka_B"] = 25.0

        # Aerodynamic parameters
        if "drag_coefficient" in parameters.keys():
            self.parameters["drag_coefficient"] = parameters["drag_coefficient"]
        else:
            self.parameters["drag_coefficient"] = 1.35  # m^2
        if "downforce_coefficient" in parameters.keys():
            self.parameters["downforce_coefficient"] = parameters[
                "downforce_coefficient"
            ]
        else:
            self.parameters["downforce_coefficient"] = 4.31  # m^2

        # Other physical parameters
        if "inertia_moment" in parameters.keys():
            self.parameters["inertia_moment"] = parameters["inertia_moment"]
        else:
            self.parameters["inertia_moment"] = 1500  # kg m^2
        if "power" in parameters.keys():
            self.parameters["power"] = parameters["power"]
        else:
            self.parameters["power"] = 462334  # W

        self.state = Racecar_State_3DOF()

        # Constant: Free fall acceleration
        self.G = 9.81  # m/s^2
        # Constant: Air density
        self.RHO = 1.225  # kg/m^3

    def weight(self) -> float:
        """
        Calculate the current weight of the racecar, including aerodynamic downforce.

        Returns:
            float: The total weight of the racecar.
        """
        speed = self.state.get_speed()

        # Short named constants
        ClA = self.parameters["downforce_coefficient"]
        mass = self.parameters["mass"]

        downforce = 0.5 * self.RHO * ClA * speed**2
        weight = mass * self.G

        return weight + downforce

    def Step(self, action: Racecar_Action) -> Racecar_State:
        """
        Simulate one time step of the dynamic model.

        Parameters:
            action (Racecar_Action): The action to apply (throttle and steering).

        Returns:
            Racecar_State: The updated state of the racecar after the simulation step.
        """
        # Short named constants
        power = self.parameters["power"]
        brake = self.parameters["brake_force"]
        mass = self.parameters["mass"]
        L2 = self.parameters["wheelbase"] / 2
        dt = self.parameters["dt"]
        CdA = self.parameters["drag_coefficient"]
        inertia = self.parameters["inertia_moment"]

        # Tire model constants
        pacD = self.parameters["pacejka_D"]
        pacC = self.parameters["pacejka_C"]
        pacB = self.parameters["pacejka_B"]

        # Get actions
        act = action.saturate(self.parameters["max_steering_angle"])

        # Short name state variables
        vel_x = self.state.velocity_x
        vel_y = self.state.velocity_y
        vel_h = self.state.velocity_heading

        # Slipping angles
        alpha_R = np.arctan2((vel_y - (L2 * vel_h)), vel_x)
        alpha_F = np.arctan2((vel_y + (L2 * vel_h)), vel_x) - act.steering

        # Pacejika Lateral Forces computation
        F_R_y = (
            -pacD * np.sin(pacC * np.arctan(pacB * alpha_R)) * self.weight() / 2
        )  # Rear wheel
        F_F_y = (
            -pacD * np.sin(pacC * np.arctan(pacB * alpha_F)) * self.weight() / 2
        )  # Front wheel

        # Force generated by engine or brakes
        engine = power / (
            self.state.get_speed() if self.state.get_speed() > 1.0 else 1.0
        )
        F_x = act.throttle * (
            engine if act.throttle > 0.0 else brake * np.sign(self.state.velocity_x)
        )

        # Maximum friction force available with current conditions
        max_friction_force = pacD * self.weight()
        F_x = np.clip(F_x, -max_friction_force, max_friction_force)

        # Drag forces computation
        F_x_aero = -0.5 * np.sign(vel_x) * CdA * self.RHO * vel_x**2
        F_y_aero = -0.5 * np.sign(vel_y) * CdA * self.RHO * vel_y**2

        # Acceleration
        a_x = ((F_x - (F_F_y * np.sin(act.steering)) + F_x_aero) / mass) + (
            vel_y * vel_h
        )
        a_y = ((F_R_y + (F_F_y * np.cos(act.steering)) + F_y_aero) / mass) - (
            vel_x * vel_h
        )

        # Heading angular acceleration
        a_h = ((F_F_y * L2 * np.cos(act.steering)) - (F_R_y * L2)) / inertia

        # Full stop condition
        if abs(a_x * dt) > abs(vel_x) and act.throttle < 0.0:
            self.state.velocity_x = 0.0
        else:
            self.state.velocity_x += a_x * dt

        self.state.velocity_y += a_y * dt

        self.state.velocity_heading += a_h * dt

        velocity = self.state.velocity()

        # Rotation matrix implementation
        cos_h = np.cos(self.state.heading)
        sin_h = np.sin(self.state.heading)
        rot_mat = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

        position_dot = np.matmul(rot_mat, velocity) * dt

        self.state.x += position_dot[0]
        self.state.y += position_dot[1]
        self.state.heading += self.state.velocity_heading * dt

        return self.state.get_Racecar_State()

    def Initialization(self, state: Racecar_State = Racecar_State()) -> None:
        """
        Define the initial condition of the racecar.

        Parameters:
            state (Racecar_State): Initial state of the racecar to set up.

        Returns:
            None
        """
        self.state = Racecar_State_3DOF(copy.copy(state))
        