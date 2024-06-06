import track
import model
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


class MPC_Tracking(track.Abstrack_tracking):
    def __init__(self, parameters: dict = {}) -> None:
        super().__init__(parameters)
        
        self.set_default_parameters(parameters, "R", [0.1, 0.1])
        self.set_default_parameters(parameters, "Q", [10, 10, 1, 1])
        
        self.algorithm = "mpc"
        
        self.warmup_control = np.zeros((2, self.PREDICTION_HORIZON), dtype=float)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        
        self.predictive_model = BicycleModel(
            device=self.device,
            parameters=self.parameters.copy(),
        )  
       
    def __cost_function(self, actions: torch.tensor):
        self.predictive_model.Initialization(self.model.state.get_Racecar_State())
        for i in range(self.parameters["prediction_horizon"]):
            act = model.Racecar_Action(actions[i*2], actions[i*2+1])
            next_state = self.predictive_model.Step(act)
            cost = next_state.quadratic_error(self.reference_states[i], self.parameters["Q"]) +\
                act.quadratic_error(model.Racecar_Action(), weight=self.parameters["R"])
        return cost
    
    def tracking_initialization(self) -> None:
        pass
        
    def tracking_termination(self) -> None:
        pass
        
    #def control_step(self) -> model.Racecar_Action:
    #    actions = torch.tensor([0.0] * self.PREDICTION_HORIZON * 2, requires_grad=True)        
    #    optimizer = optim.AdamW([actions], lr=0.1, amsgrad=True)        
    #    for iter in range(0, 100):
    #        optimizer.zero_grad()
    #        cost = self.__cost_function(actions)
    #        with torch.autograd.set_detect_anomaly(True):
    #            cost.backward()
    #        optimizer.step()
    #        
    #    return model.Racecar_Action(actions[0], actions[1])
    
    
    def control_step(self) -> model.Racecar_Action:
        accels = self.warmup_control[0].tolist()
        steering = self.warmup_control[1].tolist()
        self.predictive_model.set_controls(accels, steering)

        optimizer = optim.AdamW(self.predictive_model.parameters(), lr=0.1, amsgrad=True)

        target_x = [torch.tensor(self.reference_states[i].x).to(self.device) for i in range(0, len(self.reference_states))]
        target_y = [torch.tensor(self.reference_states[i].y).to(self.device) for i in range(0, len(self.reference_states))]
        target_vel = [torch.tensor(self.reference_states[i].speed).to(self.device) for i in range(0, len(self.reference_states))]
        target_omega = [torch.tensor(self.reference_states[i].heading).to(self.device) for i in range(0, len(self.reference_states))]
        x_0 = torch.tensor(self.model.state.x).to(self.device)
        y_0 = torch.tensor(self.model.state.y).to(self.device)
        speed_0 = torch.tensor(self.model.state.get_Racecar_State().speed).to(self.device)
        yaw_0 = torch.tensor(self.model.state.heading).to(self.device)

        for iter in range(0, 100):
            optimizer.zero_grad()

            x, y, yaw, speed = self.predictive_model(x_0, y_0, yaw_0, speed_0)
            errors_x = [(x[i] - target_x[i]) ** 2.0 for i in range(0, len(x))]
            errors_y = [(y[i] - target_y[i]) ** 2.0 for i in range(0, len(x))]
            errors_vel = [(speed[i] - target_vel[i]) ** 2.0 for i in range(0, len(x))]
            errors_omega = [(yaw[i] - target_omega[i]) ** 2.0 for i in range(0, len(x))]
            constraints = [parameter ** 2.0 for parameter in self.predictive_model.parameters()]

            loss_mse_x = torch.sum(torch.stack(errors_x, dim=0), dim=0)
            loss_mse_y = torch.sum(torch.stack(errors_y, dim=0), dim=0)
            loss_mse_vel = torch.sum(torch.stack(errors_vel, dim=0), dim=0)
            loss_mse_omega = torch.sum(torch.stack(errors_omega, dim=0), dim=0)

            loss_reg = torch.sum(torch.stack(constraints, dim=0), dim=0)
            total_loss = self.parameters['Q'][0] * loss_mse_x + \
                         self.parameters['Q'][1] * loss_mse_y + \
                         self.parameters['Q'][2] * loss_mse_vel + \
                         self.parameters['Q'][3] * loss_mse_omega + \
                         self.parameters['R'][0] * loss_reg
            total_loss.backward()
            optimizer.step()

        prediction, control = self.predictive_model.get_state()
        
        self.warmup_control = np.copy(control)
        
        action = model.Racecar_Action(control[0][0], control[1][0])

        return action

class BicycleModel(nn.Module):
    def __init__(self, device, parameters: dict):
        super().__init__()
        self.device = device     
        
        self.PREDICTION_HORIZON = parameters["prediction_horizon"]
        self.wheelbase = torch.tensor(parameters["wheelbase"]).to(device)
        self.max_steer = torch.tensor(parameters["max_steering_angle"]).to(device)
        self.dt = torch.tensor(parameters["dt"]).to(device)
        self.max_acc = torch.tensor(parameters["engine_force"]/parameters["mass"]).to(device)
        
        # State
        self.x = [torch.tensor(0.0).to(device) for i in range(0, self.PREDICTION_HORIZON)]
        self.y = [torch.tensor(0.0).to(device) for i in range(0, self.PREDICTION_HORIZON)]
        self.yaw = [torch.tensor(0.0).to(device) for i in range(0, self.PREDICTION_HORIZON)]
        self.speed = [torch.tensor(0.0).to(device) for i in range(0, self.PREDICTION_HORIZON)]

        # Control - accelerations and steering angles (front wheels angle) for all time steps
        self.accel = nn.ParameterList([nn.Parameter(torch.tensor(0.0, requires_grad=True)).to(device) for i in range(0, self.PREDICTION_HORIZON)])
        self.steering = nn.ParameterList([nn.Parameter(torch.tensor(0.0, requires_grad=True)).to(device) for i in range(0, self.PREDICTION_HORIZON)])

    def set_controls(self, accelerations, steering_angles):
        self.accel = nn.ParameterList([nn.Parameter(torch.tensor(accelerations[i], requires_grad=True).to(self.device)) for i in range(0, self.PREDICTION_HORIZON)])
        self.steering = nn.ParameterList([nn.Parameter(torch.tensor(steering_angles[i], requires_grad=True).to(self.device)) for i in range(0, self.PREDICTION_HORIZON)])

    def get_state(self):
        x_list = [self.x[i].item() for i in range(0, self.PREDICTION_HORIZON)]
        y_list = [self.y[i].item() for i in range(0, self.PREDICTION_HORIZON)]
        yaw_list = [self.yaw[i].item() for i in range(0, self.PREDICTION_HORIZON)]
        speed_list = [self.speed[i].item() for i in range(0, self.PREDICTION_HORIZON)]
        accel_list = [self.accel[i].item() for i in range(0, self.PREDICTION_HORIZON)]
        steering_list = [self.steering[i].item() for i in range(0, self.PREDICTION_HORIZON)]
        return np.array([x_list, y_list, speed_list, yaw_list]), np.array([accel_list, steering_list])
    
    def get_controls(self):
        accel_list = [self.accel[i].item() for i in range(0, self.PREDICTION_HORIZON)]
        steering_list = [self.steering[i].item() for i in range(0, self.PREDICTION_HORIZON)]
        return accel_list, steering_list

    def forward(self, start_x, start_y, start_yaw, start_speed):
        # Set initial conditions
        self.x[0] = start_x
        self.y[0] = start_y
        self.yaw[0] = start_yaw
        self.speed[0] = start_speed

        for i in range(0, self.PREDICTION_HORIZON-1):
            # Compute speed
            throttle = torch.clamp(self.accel[i], -1.0, 1.0)
            self.speed[i+1] = self.speed[i] + self.dt*throttle*self.max_acc
            
            # Clamp steering control and compute current angular velocity
            steering_angle = torch.clamp(self.steering[i], -self.max_steer, self.max_steer)
            angular_velocity = self.speed[i]*torch.tan(steering_angle)/self.wheelbase

            self.x[i+1] = self.x[i] + self.speed[i]*torch.cos(self.yaw[i])*self.dt
            self.y[i+1] = self.y[i] + self.speed[i]*torch.sin(self.yaw[i])*self.dt
            self.yaw[i+1] = self.yaw[i] + angular_velocity*self.dt
        
        return self.x, self.y, self.yaw, self.speed