import track
import model
import deepc
import numpy as np


class DEEPC_Tracking(track.Abstrack_tracking):
    def __init__(self, parameters: dict = {}) -> None:
        super().__init__(parameters)
        
        self.set_default_parameters(parameters, "N", 100)
        self.set_default_parameters(parameters, "R", [0.1, 0.1])
        self.set_default_parameters(parameters, "Q", [10, 10, 1, 1])
        self.set_default_parameters(parameters, "lambda_y", [50.0]*4)
        self.set_default_parameters(parameters, "lambda_g", 1.0)
        self.set_default_parameters(parameters, "vel_min", 5.0)
        self.set_default_parameters(parameters, "vel_max", 90.0)
        self.set_default_parameters(parameters, "seed", 1)
        
        self.parameters["n_inputs"] = 2
        self.parameters["n_outputs"] = 4
                
        self.algorithm = "deepc"        
        
        self.deepc = deepc.DeepC(self.parameters)
        self.deepc.set_opt_criteria(self.parameters.copy())
        
    def __random_experiment(self, seed):
        np.random.seed(seed)
        dataset_inputs = []
        dataset_outputs = []
        trials = 0
        
        total_length = self.INITIAL_HORIZON + self.PREDICTION_HORIZON
        
        while trials < self.parameters["N"]:
            
            initial_state = model.Racecar_State()
            initial_state.speed = np.random.uniform(
                self.parameters["vel_min"], 
                self.parameters["vel_max"],
            )
            self.model.Initialization(initial_state)
            
            action = model.Racecar_Action(
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(-self.parameters["max_steering_angle"], 
                                  self.parameters["max_steering_angle"]),
            )
            
            is_stopped = False
            
            realization_outputs = []
            realization_actions = []
            
            for t in range(0, total_length):
                realization_actions.append(action.get_numpy())
                next_state = self.model.Step(action)
                realization_outputs.append(next_state.get_numpy())
                if next_state.speed < 1.0:
                    is_stopped = True
                    break
                
                action = model.Racecar_Action(
                    np.random.uniform(-1.0, 1.0),
                    np.random.uniform(-self.parameters["max_steering_angle"], 
                                      self.parameters["max_steering_angle"]),
                )
            
            if not is_stopped:
                dataset_inputs.append(np.array(realization_actions).T)
                dataset_outputs.append(np.array(realization_outputs).T)
                trials += 1
        
        self.deepc.set_data(dataset_inputs, dataset_outputs)
        
    def tracking_initialization(self) -> None:
        self.__random_experiment(self.parameters["seed"])
        self.deepc.dataset_reformulation(self.deepc.dataset)
        
    def control_step(self) -> model.Racecar_Action:
        
        pole = self.model.state.get_Racecar_State()

        # Initial condition setup
        inputs = np.array([each.get_numpy() for each in self.past_actions]).T
        outputs = np.array([each.coordinate_change(pole).get_numpy() for each in self.past_states]).T        
        self.deepc.set_init_cond(inputs, outputs)
        
        # Reference trajectory
        reference = np.array([each.coordinate_change(pole).get_numpy() for each in self.reference_states]).T
        self.deepc.set_reference(reference)
        
        # Solve the optimization problem
        result = self.deepc.solve().T
        action = model.Racecar_Action(result[0][0], result[0][1])
        
        return action
        
    def tracking_termination(self) -> None:
        pass
        
        
        
        
        
        
        
        
        
        
        