from abc import ABC, abstractmethod
import model
import numpy as np
import copy
import csv

class Tracking_state:
    def __init__(self) -> None:
        self.state = "Initialized"
        self.error = ""
        
    def set_error(self, error: str) -> None:
        self.state = "Error"
        self.error = error
        
    def set_state(self, state: str) -> None:
        if state != "Error":
            self.state = state
        
    def is_error(self) -> bool:
        return self.state == "Error"

class Abstrack_tracking(ABC):
    def __init__(self, parameters: dict = {}) -> None:
        super().__init__()
        
        self.past_states = []
        self.reference_states = []
        
        if True:
            # Initialize the parameters
            self.parameters = {}
            if "model" in parameters.keys():
                self.parameters["model"] = parameters["model"]
            else:
                self.parameters["model"] = 'dynamic'
                
            if "track" in parameters.keys():
                self.parameters["track"] = parameters["track"]
            else:
                self.parameters["track"] = 'Lissajous'
                
            if 'Lissajous_a' in parameters.keys():
                self.parameters['Lissajous_a'] = parameters['Lissajous_a']
            else:
                self.parameters['Lissajous_a'] = 1

            if 'Lissajous_b' in parameters.keys():
                self.parameters['Lissajous_b'] = parameters['Lissajous_b']
            else:
                self.parameters['Lissajous_b'] = 2

            if 'Lissajous_phase' in parameters.keys():
                self.parameters['Lissajous_phase'] = parameters['Lissajous_phase']
            else:
                self.parameters['Lissajous_phase'] = 0.0

            if 'Lissajous_radius' in parameters.keys():
                self.parameters['Lissajous_radius'] = parameters['Lissajous_radius']
            else:
                self.parameters['Lissajous_radius'] = 100.0

            if 'Lissajous_circle_time' in parameters.keys():
                self.parameters['Lissajous_circle_time'] = parameters['Lissajous_circle_time']
            else:
                self.parameters['Lissajous_circle_time'] = 5000
                
            if 'track_upper_bound' in parameters.keys():
                self.parameters['track_upper_bound'] = parameters['track_upper_bound']
            else:
                self.parameters['track_upper_bound'] = 100.0

            if 'track_lower_bound' in parameters.keys():
                self.parameters['track_lower_bound'] = parameters['track_lower_bound']
            else:
                self.parameters['track_lower_bound'] = 0.0
                
            if 'initial_horizon' in parameters.keys():
                self.parameters['initial_horizon'] = parameters['initial_horizon']
            else:
                self.parameters['initial_horizon'] = 1
                
            if 'prediction_horizon' in parameters.keys():
                self.parameters['prediction_horizon'] = parameters['prediction_horizon']
            else:
                self.parameters['prediction_horizon'] = 1
                
            if 'max_tracking_error' in parameters.keys():
                self.parameters['max_tracking_error'] = parameters['max_tracking_error']
            else:
                self.parameters['max_tracking_error'] = 9999.9
                
            if 'dt' in parameters.keys():
                self.parameters['dt'] = parameters['dt']
            else:
                self.parameters['dt'] = 0.01
            
        # Horizon parameters
        self.INITIAL_HORIZON = self.parameters['initial_horizon']
        self.PREDICTION_HORIZON = self.parameters['prediction_horizon']
            
        # Initialize the model    
        if self.parameters["model"] == 'dynamic':
            self.model = model.Dynamic_model(parameters)
        elif self.parameters["model"] == 'kinematic':
            self.model = model.Kinematic_model(parameters)
            
        # Initialize the trajectory
        if self.parameters["track"] == 'Lissajous':
            trajectory = self.trajectory_generation(
                circle_time=self.parameters['Lissajous_circle_time'],
                radius=self.parameters['Lissajous_radius'],
                a=self.parameters['Lissajous_a'],
                b=self.parameters['Lissajous_b'],
                delta=self.parameters['Lissajous_phase'],
            )
        else:
            trajectory = self.import_trajectory(self.parameters['track'])
        
        trajectory_cutted = self.cut_trajectory(
            trajectory,
            self.parameters['track_lower_bound'],
            self.parameters['track_upper_bound'],
        )
        
        self.set_trajectory(trajectory_cutted)
        
        self.state = Tracking_state()

    def trajectory_generation(
        self,
        circle_time: int = 1000,
        radius: float = 25,
        a: float = 2,
        b: float = 1,
        delta: float = 0.0
    ) -> np.ndarray:
        """
        Generate a trajectory for a point moving along an Lissajous figure.

        Args:
            circle_time (int): The number of time steps to complete a full circle.
            radius (float): The radius of the Lissajous figure.
            a (float): Scaling factor for sine function (controls movement along x-axis).
            b (float): Scaling factor for cosine function (controls movement along y-axis).
            delta (float): The phase angle of Lissajous figure (controls rotation of Lissajous figure).

        Returns:
            trajectory (np.ndarray): A list containing two lists representing x and y coordinates of the trajectory.

        Note:
            This function does not return any value explicitly. It modifies the `trajectory` variable in-place.

        """
        trajectory = [[0.0] * circle_time, [0.0] *
                      circle_time]  # Initialize the trajectory with zeros
        for t in range(0, circle_time):  # Iterate over each time step
            time_var = t * 2.0 * np.pi / circle_time  # Calculate time in radians
            # Calculate x and y coordinates using sine and cosine functions scaled by a and b respectively
            trajectory[0][t] = radius * np.sin(a * time_var + delta)
            trajectory[1][t] = radius * np.sin(b * time_var)
        return np.array(trajectory)
        
    def set_trajectory(self, trajectory):
        if len(trajectory) != 2:  # X and Y
            self.state.set_error('Reference trajectory must content X and Y')
        if len(trajectory[0]) < self.INITIAL_HORIZON + self.PREDITION_HORIZON:
            self.state.set_error('Reference must content at least finish_length samples')
        if len(trajectory[1]) < self.INITIAL_HORIZON + self.PREDITION_HORIZON:
            self.state.set_error('Reference must content at least finish_length samples')
        if len(trajectory[0]) != len(trajectory[1]):
            self.state.set_error('Reference must content the same samples for X and Y')

        self.trajectory = []
        
        for i in range(1, trajectory[0]):
            dx = trajectory[0][i] - trajectory[0][i-1]
            dy = trajectory[1][i] - trajectory[1][i-1]
            velocity = np.sqrt(dx ** 2 + dy ** 2) / self.parameters['dt']
            theta = np.arctan2(dy, dx)
            if i == 1:
                state = model.Racecar_State(
                    trajectory[0][1], 
                    trajectory[1][1], 
                    velocity, 
                    theta
                )
                self.trajectory.append(copy.copy(state))
                state.x = trajectory[0][0]
                state.y = trajectory[1][0]
                self.trajectory.append(copy.copy(state))
            else:
                while theta - self.trajectory[-1].heading > np.pi/2:
                    theta = theta - 2 * np.pi
                while theta - self.trajectory[-1].heading < -np.pi/2:
                    theta = theta + 2 * np.pi
                state = model.Racecar_State(
                    trajectory[0][i], 
                    trajectory[1][i], 
                    velocity, 
                    theta
                )
                self.trajectory.append(copy.copy(state))
    
    def cut_trajectory(
        self, 
        trajectory, 
        lower: float = 0.0,
        upper: float = 100.0
    ) -> np.ndarray:
        total_len = float(len(trajectory[0]))
        lower_int = int(lower*total_len/100.0)
        upper_int = int(upper*total_len/100.0)
        trajectory_out = trajectory.T[lower_int:upper_int].T
        return trajectory_out
    
    def import_trajectory(self, file_path: str) -> np.ndarray:
        """
        Import a trajectory from a file.

        Args:
            file_path (str): The path to the file containing the trajectory.

        Returns:
            trajectory (np.ndarray): A list containing two lists representing x and y coordinates of the trajectory.

        Note:
            This function does not return any value explicitly. It modifies the `trajectory` variable in-place.

        """
        lol = []
        f = open(file_path, 'rt')
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            lol.append([float(i) for i in row])
        trajectory = np.array(lol).T
        return trajectory
    
    def winshift(self, vector: list, new_val) -> list:
        new_vector = copy.copy(vector[1:])
        new_vector.append(new_val)
        return new_vector
    
    @abstractmethod
    def tracking_initialization(self) -> None:
        pass
    
    @abstractmethod
    def control_step(self, Initial_conditions) -> model.Racecar_Action:
        pass
    
    @abstractmethod
    def tracking_termination(self) -> model.Racecar_Action:
        pass
    
    def trajectory_tracking(self):
        
        self.state.set_state("Tracking")
        
        self.past_states = copy.copy(self.trajectory[:self.INITIAL_HORIZON])
        self.model.Initialization(self.past_states[-1])        
        self.reference_states = copy.copy(self.trajectory[self.INITIAL_HORIZON:self.INITIAL_HORIZON + self.PREDICTION_HORIZON])
        
        self.tracking_initialization()
                
        for i in range(self.INITIAL_HORIZON, len(self.trajectory[0]) - self.PREDICTION_HORIZON):
            
            action = self.control_step()
            if self.state.is_error():
                break
            racecar_step_after = self.model.Step(action)
            self.past_states = self.winshift(self.past_states, racecar_step_after)
            self.reference_states = self.winshift(self.reference_states, self.trajectory[i+self.PREDICTION_HORIZON])
                        
        self.state.set_state("Postprocessing")
            
        self.tracking_termination()
        
        self.state.set_state("Finished")
            
