from abc import ABC, abstractmethod
import model
import numpy as np
import copy
import csv
import json
from termcolor import colored
import time
import base64
import hashlib
import re
import os

class Tracking_state:
    def __init__(self) -> None:
        self.state = "Initialized"
        self.error = ""
        
    def set_error(self, error: str) -> None:
        self.state = "Error"
        self.error = error
        
    def set_state(self, state: str) -> None:
        if self.state != "Error":
            self.state = state
        
    def is_error(self) -> bool:
        return self.state == "Error"
    

class Abstrack_tracking(ABC):
    def set_default_parameters(self, dict_in: dict, name: str, value) -> None:
        if name in dict_in.keys():
                self.parameters[name] = dict_in[name]
        else:
            self.parameters[name] = value
    
    def __init__(self, parameters: dict = {}) -> None:        
        super().__init__()
        
        self.past_states = []
        self.past_actions = []
        self.reference_states = []
        self.algorithm = None
        
        if True:
            # Initialize the parameters
            self.parameters = {}
            self.set_default_parameters(parameters, 'model', 'dynamic')
            self.set_default_parameters(parameters, 'track', 'Lissajous')
            self.set_default_parameters(parameters, 'Lissajous_a', 1)
            self.set_default_parameters(parameters, 'Lissajous_b', 2)
            self.set_default_parameters(parameters, 'Lissajous_phase', 0.0)
            self.set_default_parameters(parameters, 'Lissajous_radius', 100.0)
            self.set_default_parameters(parameters, 'Lissajous_circle_time', 5000)
            self.set_default_parameters(parameters, 'track_upper_bound', 100.0)
            self.set_default_parameters(parameters, 'track_lower_bound', 0.0)
            self.set_default_parameters(parameters, 'initial_horizon', 1)
            self.set_default_parameters(parameters, 'prediction_horizon', 1)
            self.set_default_parameters(parameters, 'max_tracking_error', 9999.9)
            self.set_default_parameters(parameters, 'dt', 0.01)
            self.set_default_parameters(parameters, 'print_out', 'Computation')
            self.set_default_parameters(parameters, 'save_folder', 'results')
            
        # Horizon parameters
        self.INITIAL_HORIZON = self.parameters['initial_horizon']
        self.PREDICTION_HORIZON = self.parameters['prediction_horizon']
            
        # Initialize the model    
        if self.parameters["model"] == 'dynamic':
            self.model = model.Dynamic_model(parameters)
        elif self.parameters["model"] == 'kinematic':
            self.model = model.Kinematic_model(parameters)
            
        self.parameters.update(self.model.parameters)
            
        # Initialize the trajectory
        if self.parameters["track"] == 'Lissajous':
            trajectory = self.__trajectory_generation(
                circle_time=self.parameters['Lissajous_circle_time'],
                radius=self.parameters['Lissajous_radius'],
                a=self.parameters['Lissajous_a'],
                b=self.parameters['Lissajous_b'],
                delta=self.parameters['Lissajous_phase'],
            )
        else:
            trajectory = self._import_trajectory(self.parameters['save_folder']+'/'+self.parameters['track'])
        
        trajectory_cutted = self.__cut_trajectory(
            trajectory,
            self.parameters['track_lower_bound'],
            self.parameters['track_upper_bound'],
        )
        
        self._set_trajectory(trajectory_cutted)
        
        self.state = Tracking_state()

    def __trajectory_generation(
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
        
    def _set_trajectory(self, trajectory):
        if len(trajectory) != 2:  # X and Y
            self.state.set_error('Reference trajectory must content X and Y')
        if len(trajectory[0]) < self.INITIAL_HORIZON + self.PREDICTION_HORIZON:
            self.state.set_error('Reference must content at least finish_length samples')
        if len(trajectory[1]) < self.INITIAL_HORIZON + self.PREDICTION_HORIZON:
            self.state.set_error('Reference must content at least finish_length samples')
        if len(trajectory[0]) != len(trajectory[1]):
            self.state.set_error('Reference must content the same samples for X and Y')

        self.trajectory = []
        
        for i in range(1, len(trajectory[0])):
            dx = trajectory[0][i] - trajectory[0][i-1]
            dy = trajectory[1][i] - trajectory[1][i-1]
            velocity = np.sqrt(dx ** 2 + dy ** 2) / self.parameters['dt']
            theta = np.arctan2(dy, dx)
            if i == 1:
                state = model.Racecar_State(
                    trajectory[0][0], 
                    trajectory[1][0], 
                    velocity, 
                    theta
                )
                self.trajectory.append(state.copy())
                state.x = trajectory[0][1]
                state.y = trajectory[1][1]
                self.trajectory.append(state.copy())
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
                self.trajectory.append(state.copy())
    
    def __cut_trajectory(
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
    
    def _import_trajectory(self, file_path: str) -> np.ndarray:
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
    
    def __winshift(self, vector: list, new_val) -> list:
        new_vector = copy.copy(vector[1:])
        new_vector.append(new_val)
        return new_vector
    
    @abstractmethod
    def tracking_initialization(self) -> None:
        pass
    
    @abstractmethod
    def control_step(self) -> model.Racecar_Action:
        pass
    
    @abstractmethod
    def tracking_termination(self) -> None:
        pass
    
    def __tracking_error(self, result: list) -> float:
        bank = 0.0
        shift = self.INITIAL_HORIZON + 1
        for i in range(0, len(result)):
            dx = result[i].x-self.trajectory[i+shift].x
            dy = result[i].y-self.trajectory[i+shift].y
            dist = np.sqrt(dx ** 2 + dy ** 2)
            bank += dist
        return bank/len(result)
    
    def __restore_result(self, file_name: str) -> list:
        result = []
        file_path = self.parameters['save_folder'] + '/' + file_name
        np_array = np.load(file_path)
        for each in np_array:
            state = model.Racecar_State(each[0], each[1], each[2], each[3])
            result.append(state)
        return result
    
    def __hash_generator(self) -> str:            
        hash_dict = copy.copy(self.parameters)
        hash_dict.pop('save_folder', None)
        hash_dict.pop('print_out', None)
        hash_dict['algorithm'] = self.algorithm
        if hash_dict['track'] != 'Lissajous':
            hash_dict.pop('Lissajous_a', None)
            hash_dict.pop('Lissajous_b', None)
            hash_dict.pop('Lissajous_phase', None)
            hash_dict.pop('Lissajous_radius', None)
            hash_dict.pop('Lissajous_circle_time', None)
        summary = base64.b64encode(hashlib.md5(
            str(hash_dict).encode()).digest()).replace(b'=', b'a').decode()
        id = f"{self.parameters['track']}-{self.algorithm}-{self.parameters['prediction_horizon']}-{summary}"
        s = str(id).strip().replace(" ", "_")
        file_name = re.sub(r"(?u)[^-\w.]", "", s)
        return file_name

    def get_hash(self) -> str:
            return self.__hash_generator()
    
    def __save_result(self, result: list, rss: float, time: float) -> None:
        # Save the result to numpy file and self.parameters to json file
        
        file_name = self.__hash_generator()
        file_path = self.parameters['save_folder'] + '/' + file_name
        os.makedirs(self.parameters['save_folder'], exist_ok=True)
        numpy_array = []
        for each in result:
            numpy_array.append(each.get_numpy())
        np.save(file_path, np.array(numpy_array))
        
        metadata = {
            'algorithm': self.algorithm,
            'state': self.state.state,
            'rss': rss,
            'time': time,
            'id': file_name,
        }
        
        if self.state.is_error():
            metadata['error'] = self.state.error
        
        local_parameters = copy.copy(self.parameters)
        local_parameters.update(metadata)
        
        with open(file_path + '.json', 'w+') as f:
            json.dump(local_parameters, f)
            
    def __is_simulation_exist(self) -> bool:
        file_name = self.__hash_generator()
        file_path = self.parameters['save_folder'] + '/' + file_name
        return os.path.exists(file_path + '.json')   
    
    def get_rss(self) -> float:
        if self.__is_simulation_exist():
            file_name = self.__hash_generator()
            file_path = self.parameters['save_folder'] + '/' + file_name + '.json'
            with open(file_path, 'r') as f:
                metadata = json.load(f)
            return metadata['rss'], metadata['state']
        else:
            return None
    
    def trajectory_tracking(self) -> list:
        
        if self.__is_simulation_exist():
            if self.parameters['print_out'] != 'Nothing':
                print(colored('Trajectory tracking: ', 'green'), colored('Suspended', 'white'))
                print(colored('Simulation is already done!', 'white'))
                result = self.__restore_result(self.__hash_generator() + '.npy')
                self.rss, state = self.get_rss()
                if state == "Error":
                    self.state.set_error('Simulation is already done with error')
                else:
                    self.state.set_state(state)
            return result
        
        self.state.set_state("Tracking")
        
        self.past_states = copy.copy(self.trajectory[:self.INITIAL_HORIZON])
        for _ in range(self.INITIAL_HORIZON):
            self.past_actions.append(model.Racecar_Action(0.0, 0.0))
            
        self.reference_states = copy.copy(self.trajectory[self.INITIAL_HORIZON:self.INITIAL_HORIZON + self.PREDICTION_HORIZON])
        
        self.tracking_initialization()
                
        self.model.Initialization(self.past_states[-1])    
                
        result = []
        
        total_time = len(self.trajectory) - self.INITIAL_HORIZON - self.PREDICTION_HORIZON
        
        percent_counter = 1
        
        start_time = time.time()
                
        for t in range(self.INITIAL_HORIZON, len(self.trajectory) - self.PREDICTION_HORIZON):            
            
            percent = 100.0 * (t - self.INITIAL_HORIZON) / total_time
            if percent > percent_counter:
                percent_counter += 1
                if self.parameters['print_out'] != 'Nothing':
                    print(colored('Trajectory tracking: ', 'yellow'), colored(percent_counter - 1, 'white'), colored('%', 'yellow'))
                    current_time = (101 - percent_counter) * (time.time() - start_time) / percent_counter
                    print(colored('Estimated time: ', 'red'), colored(current_time, 'white'), colored(' sec. left', 'red'))
                        
            action = self.control_step()
            
            if self.state.is_error():
                break
            
            if self.parameters['print_out'] == 'Everything':
                print(colored('Action: ', 'green'), colored(action, 'white'))
            
            racecar_step_after = self.model.Step(action)

            deviation = self.reference_states[0].get_numpy() - racecar_step_after.get_numpy()
            if np.sqrt((deviation[0] ** 2 + deviation[1] ** 2)) > self.parameters['max_tracking_error']:
                self.state.set_error('Tracking error is too high')
                break
            
            if self.parameters['print_out'] == 'Everything':
                print(colored('State after: ', 'green'), colored(racecar_step_after, 'white'))
            
            result.append(racecar_step_after.copy())
            self.past_states = self.__winshift(self.past_states, racecar_step_after)
            self.past_actions = self.__winshift(self.past_actions, action)
            self.reference_states = self.__winshift(self.reference_states, self.trajectory[t+self.PREDICTION_HORIZON])
                        
        self.state.set_state("Postprocessing")
        
        exec_time = time.time()-start_time
        
        error_per = self.__tracking_error(result)
        
        self.rss = error_per
        
        if self.parameters['print_out'] != 'Nothing':
            if self.state.is_error():
                print(colored('Trajectory tracking: ', 'green') + colored('Failed', 'red'))
                print(colored('Error: ', 'red'), colored(self.state.error, 'white'))
            else:
                print(colored('Trajectory tracking: Succeded', 'green'))
            print(colored('Accumulated tracking error per point:', 'green'), colored('{:.2f} m.'.format(error_per), 'white'))
            print(colored('Execution time:', 'green'), colored('{:.2f} s.'.format(exec_time), 'white'))
            print(colored('Averege responce time:', 'green'), colored('{:.2f} s.'.format(exec_time/total_time), 'white'))
            
        self.tracking_termination()
        
        self.state.set_state("Finished")
            
        self.__save_result(result, error_per, exec_time)
            
        return result
