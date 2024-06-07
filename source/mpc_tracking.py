import track
import model
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


class MPC_Tracking(track.Abstrack_tracking):
    """
    Model Predictive Control (MPC) tracking algorithm implementation.

    Args:
        parameters (dict): Dictionary containing MPC parameters.

    Attributes:
        algorithm (str): Name of the algorithm, set to 'mpc'.
        warmup_control (np.array): Array storing warm-up control values.
        device (torch.device): Device for computation (GPU if available, else CPU).
        predictive_model (BicycleModel): Instance of the predictive model.
    """

    def __init__(self, parameters: dict = {}) -> None:
        super().__init__(parameters)

        self.set_default_parameters(parameters, "R", [0.1, 0.1])
        self.set_default_parameters(parameters, "Q", [10, 10, 1, 1])

        self.algorithm = "mpc"

        self.warmup_control = np.zeros((2, self.PREDICTION_HORIZON), dtype=float)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.predictive_model = BicycleModel(
            device=self.device,
            parameters=self.parameters.copy(),
        )

    def tracking_initialization(self) -> None:
        """
        Initialization step for the tracking algorithm.
        """
        pass

    def tracking_termination(self) -> None:
        """
        Termination step for the tracking algorithm.
        """
        pass

    def control_step(self) -> model.Racecar_Action:
        """
        Perform a control step based on the current state and reference state.

        Returns:
            model.Racecar_Action: An object containing the steering and throttle values.
        """
        accels = self.warmup_control[0].tolist()
        steering = self.warmup_control[1].tolist()
        self.predictive_model.set_controls(accels, steering)

        optimizer = optim.AdamW(
            self.predictive_model.parameters(), lr=0.1, amsgrad=True
        )

        target_x = [
            torch.tensor(self.reference_states[i].x).to(self.device)
            for i in range(0, len(self.reference_states))
        ]
        target_y = [
            torch.tensor(self.reference_states[i].y).to(self.device)
            for i in range(0, len(self.reference_states))
        ]
        target_vel = [
            torch.tensor(self.reference_states[i].speed).to(self.device)
            for i in range(0, len(self.reference_states))
        ]
        target_omega = [
            torch.tensor(self.reference_states[i].heading).to(self.device)
            for i in range(0, len(self.reference_states))
        ]
        x_0 = torch.tensor(self.model.state.x).to(self.device)
        y_0 = torch.tensor(self.model.state.y).to(self.device)
        speed_0 = torch.tensor(self.model.state.get_Racecar_State().speed).to(
            self.device
        )
        yaw_0 = torch.tensor(self.model.state.heading).to(self.device)

        relative_cost_threshold = 0.000001  # Stop if improvement is less than 1%
        previous_loss = float("inf")

        for iter in range(0, 200):
            optimizer.zero_grad()

            x, y, yaw, speed = self.predictive_model(x_0, y_0, yaw_0, speed_0)
            errors_x = [(x[i] - target_x[i]) ** 2.0 for i in range(0, len(x))]
            errors_y = [(y[i] - target_y[i]) ** 2.0 for i in range(0, len(x))]
            errors_vel = [(speed[i] - target_vel[i]) ** 2.0 for i in range(0, len(x))]
            errors_omega = [(yaw[i] - target_omega[i]) ** 2.0 for i in range(0, len(x))]
            constraints = [
                parameter**2.0 for parameter in self.predictive_model.parameters()
            ]

            loss_mse_x = torch.sum(torch.stack(errors_x, dim=0), dim=0)
            loss_mse_y = torch.sum(torch.stack(errors_y, dim=0), dim=0)
            loss_mse_vel = torch.sum(torch.stack(errors_vel, dim=0), dim=0)
            loss_mse_omega = torch.sum(torch.stack(errors_omega, dim=0), dim=0)

            loss_reg = torch.sum(torch.stack(constraints, dim=0), dim=0)
            total_loss = (
                self.parameters["Q"][0] * loss_mse_x
                + self.parameters["Q"][1] * loss_mse_y
                + self.parameters["Q"][2] * loss_mse_vel
                + self.parameters["Q"][3] * loss_mse_omega
                + self.parameters["R"][0] * loss_reg
            )

            improvement = (previous_loss - total_loss.item()) / previous_loss
            if improvement < relative_cost_threshold:
                break

            previous_loss = total_loss.item()

            total_loss.backward()
            optimizer.step()

        _, control = self.predictive_model.get_state()

        self.warmup_control = np.copy(control)

        action = model.Racecar_Action(control[0][0], control[1][0])

        return action


class BicycleModel(nn.Module):
    """
    Bicycle model for MPC predictive control.

    Args:
        device (torch.device): Device for computation.
        parameters (dict): Dictionary containing model parameters.

    Attributes:
        wheelbase (torch.Tensor): Wheelbase of the vehicle.
        max_steer (torch.Tensor): Maximum steering angle.
        dt (torch.Tensor): Time step.
        max_acc (torch.Tensor): Maximum acceleration.
        x, y, yaw, speed (list of torch.Tensor): State variables.
        accel, steering (nn.ParameterList): Control inputs.
    """

    def __init__(self, device, parameters: dict):
        super().__init__()
        self.device = device

        self.PREDICTION_HORIZON = parameters["prediction_horizon"]
        self.wheelbase = torch.tensor(parameters["wheelbase"]).to(device)
        self.max_steer = torch.tensor(parameters["max_steering_angle"]).to(device)
        self.dt = torch.tensor(parameters["dt"]).to(device)
        self.max_acc = torch.tensor(parameters["engine_force"] / parameters["mass"]).to(
            device
        )

        # State
        self.x = [
            torch.tensor(0.0).to(device) for _ in range(0, self.PREDICTION_HORIZON)
        ]
        self.y = [
            torch.tensor(0.0).to(device) for _ in range(0, self.PREDICTION_HORIZON)
        ]
        self.yaw = [
            torch.tensor(0.0).to(device) for _ in range(0, self.PREDICTION_HORIZON)
        ]
        self.speed = [
            torch.tensor(0.0).to(device) for _ in range(0, self.PREDICTION_HORIZON)
        ]

        # Control - accelerations and steering angles (front wheels angle) for all time steps
        self.accel = nn.ParameterList(
            [
                nn.Parameter(torch.tensor(0.0, requires_grad=True)).to(device)
                for _ in range(0, self.PREDICTION_HORIZON)
            ]
        )
        self.steering = nn.ParameterList(
            [
                nn.Parameter(torch.tensor(0.0, requires_grad=True)).to(device)
                for _ in range(0, self.PREDICTION_HORIZON)
            ]
        )

    def set_controls(self, accelerations, steering_angles):
        """
        Set control inputs.

        Args:
            accelerations (list): List of acceleration values.
            steering_angles (list): List of steering angle values.

        Returns:
            None
        """
        self.accel = nn.ParameterList(
            [
                nn.Parameter(
                    torch.tensor(accelerations[i], requires_grad=True).to(self.device)
                )
                for i in range(0, self.PREDICTION_HORIZON)
            ]
        )
        self.steering = nn.ParameterList(
            [
                nn.Parameter(
                    torch.tensor(steering_angles[i], requires_grad=True).to(self.device)
                )
                for i in range(0, self.PREDICTION_HORIZON)
            ]
        )

    def get_state(self):
        """
        Get the current state.

        Returns:
            tuple: Tuple containing arrays of state variables (x, y, speed, yaw) and control inputs (accel, steering).
        """
        x_list = [self.x[i].item() for i in range(0, self.PREDICTION_HORIZON)]
        y_list = [self.y[i].item() for i in range(0, self.PREDICTION_HORIZON)]
        yaw_list = [self.yaw[i].item() for i in range(0, self.PREDICTION_HORIZON)]
        speed_list = [self.speed[i].item() for i in range(0, self.PREDICTION_HORIZON)]
        accel_list = [self.accel[i].item() for i in range(0, self.PREDICTION_HORIZON)]
        steering_list = [
            self.steering[i].item() for i in range(0, self.PREDICTION_HORIZON)
        ]
        return np.array([x_list, y_list, speed_list, yaw_list]), np.array(
            [accel_list, steering_list]
        )

    def get_controls(self):
        """
        Get control inputs.

        Returns:
            tuple: Tuple containing lists of acceleration and steering angle values.
        """
        accel_list = [self.accel[i].item() for i in range(0, self.PREDICTION_HORIZON)]
        steering_list = [
            self.steering[i].item() for i in range(0, self.PREDICTION_HORIZON)
        ]
        return accel_list, steering_list

    def forward(self, start_x, start_y, start_yaw, start_speed):
        """
        Perform forward prediction of the model.

        Args:
            start_x (torch.Tensor): Initial x-coordinate.
            start_y (torch.Tensor): Initial y-coordinate.
            start_yaw (torch.Tensor): Initial yaw angle.
            start_speed (torch.Tensor): Initial speed.

        Returns:
            tuple: Tuple containing arrays of predicted state variables (x, y, yaw, speed).
        """
        # Set initial conditions
        self.x[0] = start_x
        self.y[0] = start_y
        self.yaw[0] = start_yaw
        self.speed[0] = start_speed

        for i in range(0, self.PREDICTION_HORIZON - 1):
            # Compute speed
            throttle = torch.clamp(self.accel[i], -1.0, 1.0)
            self.speed[i + 1] = self.speed[i] + self.dt * throttle * self.max_acc

            # Clamp steering control and compute current angular velocity
            steering_angle = torch.clamp(
                self.steering[i], -self.max_steer, self.max_steer
            )
            angular_velocity = (
                self.speed[i] * torch.tan(steering_angle) / self.wheelbase
            )

            # Update state variables
            self.x[i + 1] = self.x[i] + self.speed[i] * torch.cos(self.yaw[i]) * self.dt
            self.y[i + 1] = self.y[i] + self.speed[i] * torch.sin(self.yaw[i]) * self.dt
            self.yaw[i + 1] = self.yaw[i] + angular_velocity * self.dt

        # Return predicted state variables
        return self.x, self.y, self.yaw, self.speed
