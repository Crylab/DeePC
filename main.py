#!/usr/bin/env python
# This line indicates that the script should be run using the Python interpreter.

import sys

# Importing the sys module to manipulate the Python runtime environment.

sys.path.insert(1, "source/")
# Inserting 'source/' at position 1 in the sys.path list, to allow importing modules from this directory.

import deepc_tracking
import pid_tracking
import mpc_tracking
import graph
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib import cm

# Importing necessary modules for tracking algorithms, graph plotting, numerical operations, parallel processing, and visualization.

viridis = cm.get_cmap("viridis", 12)
# Getting the 'viridis' colormap with 12 discrete colors from the matplotlib colormap library.


def run_experiment(dict_in: dict):
    """
    Run a single experiment based on the specified algorithm.

    Arguments:
    dict_in -- dictionary containing configuration parameters, including the algorithm to use.

    Returns:
    rss -- Residual Sum of Squares (RSS) of the tracking deviation.
    """
    if dict_in["algorithm"] == "pid":
        obj = pid_tracking.PID_Tracking(dict_in)
    elif dict_in["algorithm"] == "deepc":
        obj = deepc_tracking.DEEPC_Tracking(dict_in)
    elif dict_in["algorithm"] == "mpc":
        obj = mpc_tracking.MPC_Tracking(dict_in)
    obj.trajectory_tracking()
    rss = obj.rss
    return rss


def run_parallel(dict_in: dict, list_in: list, var_param: str, seeds: bool = False):
    """
    Run experiments in parallel with varying parameters and optionally different seeds.

    Arguments:
    dict_in -- dictionary containing configuration parameters.
    list_in -- list of parameter values to vary.
    var_param -- the parameter to vary.
    seeds -- boolean indicating whether to run experiments with different random seeds (default: False).

    Returns:
    average_list -- list of average RSS values for each parameter setting.
    sigma_list -- list of standard deviations of RSS values for each parameter setting.
    """
    param_list = []
    for pac in list_in:
        local_param = dict_in.copy()
        local_param[var_param] = pac
        if seeds:
            for seed in range(30):
                local_local_param = local_param.copy()
                local_local_param["seed"] = seed
                param_list.append(local_local_param)
        else:
            param_list.append(local_param)

    # Creating a pool of worker processes to run the experiments in parallel.
    pool = multiprocessing.Pool()
    execution_result = pool.map(run_experiment, param_list)
    pool.close()
    pool.join()

    # Calculating average and standard deviation of the results.
    average_list = []
    sigma_list = []
    counter = 0
    for _ in range(len(list_in)):
        if seeds:
            seed_list = []
            for seed in range(30):
                seed_list.append(execution_result[counter])
                counter += 1
            average = sum(seed_list) / len(seed_list)
            average_list.append(average)
            seed_average_nd = np.array(seed_list)
            sigma = np.sqrt(
                sum(np.square(seed_average_nd - average)) / seed_average_nd.size
            )
            sigma_list.append(sigma)
        else:
            average_list.append(execution_result[counter])
            counter += 1
            sigma_list.append(0)

    return average_list, sigma_list


def plot_line_shadow(list_in: list, res: tuple, linetype: str, color, name: str):
    """
    Plot a line with a shadow representing the standard deviation.

    Arguments:
    list_in -- list of x-axis values.
    res -- tuple containing two lists: average values and standard deviations.
    linetype -- line style for the plot.
    color -- color for the plot.
    name -- label for the plot.

    Returns:
    None
    """
    plt.plot(list_in, res[0], label=name, color=color, linestyle=linetype)
    plt.fill_between(
        list_in,
        [res[0][ogni] - res[1][ogni] for ogni in range(len(list_in))],
        [res[0][ogni] + res[1][ogni] for ogni in range(len(list_in))],
        alpha=0.3,
        color=color,
    )


def nice_abu_dhabi_picture():
    """
    Generate a visual representation of different tracking algorithms on the Abu Dhabi circuit.

    Arguments:
    None

    Returns:
    None
    """
    # PID part
    parameters = {
        "heading_loop": [10 ** 0.389, 0.0, 0.0],
        "velocity_loop": [10 ** 0.8621, 0.0, 0.0],
        "direction_loop": [10 ** -0.5181, 0.0, 0.0],
        "distance_loop": [10 ** -1.3539, 0.0, 0.0],
        "track": "ABU_F2.csv",
        "pacejka_D": 2.0,
    }
    obj = pid_tracking.PID_Tracking(parameters)
    pid_result = obj.trajectory_tracking()

    # DEEPC part
    parameters = {
        "lambda_g": 5.0,
        "lambda_y": [2e2] * 4,
        "N": 100,
        "track": "ABU_F2.csv",
        "Q": [1, 1, 1, 100],
        "R": [0.1, 0.1],
        "prediction_horizon": 8,
        "track_upper_bound": 100.0,
        "pacejka_D": 2.0,
    }
    obj = deepc_tracking.DEEPC_Tracking(parameters)
    deepc_result = obj.trajectory_tracking()

    # MPC part
    parameters = {
        "track": "ABU_F2.csv",
        "Q": [1, 1, 1, 100],
        "R": [0.1, 0.1],
        "prediction_horizon": 8,
        "track_upper_bound": 100.0,
        "pacejka_D": 2.0,
    }
    obj = mpc_tracking.MPC_Tracking(parameters)
    mpc_result = obj.trajectory_tracking()

    visual_params = {
        "name": "Abu Dhabi circuit",
        "xmin": -1000,
        "ymin": -1000,
        "xmax": 1000,
        "ymax": 1000,
        "vehicle_length": 5,
        "vehicle_width": 2,
    }

    visual = graph.graph_compete(visual_params)
    visual.add_state_path(deepc_result, "blue", 0, "DeePC")
    visual.add_state_landscape(obj.trajectory, "--")
    visual.add_state_path(mpc_result, "red", 250, "MPC")
    visual.add_state_path(pid_result, "black", 500, "PID")
    visual.compression(10)
    visual.transpose()
    visual.generate_pic_at("img/Nice_abu_dhabi.pdf", 80.0)


def abu_dhabi():
    """
    Run experiments on the Abu Dhabi track and plot results.

    Arguments:
    None

    Returns:
    None
    """
    list_in = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    fig, ax = plt.subplots(figsize=(7, 4))

    pid_parameters = {
        "heading_loop": [2.0, 0.0, 0.0],
        "velocity_loop": [3.0, 0.0, 0.0],
        "direction_loop": [0.1, 0.0, 0.0],
        "distance_loop": [0.1, 0.0, 0.0],
        "algorithm": "pid",
        "track": "ABU_F2.csv",
    }
    pid_parameters_tunned = {
        "heading_loop": [10 ** 0.389, 0.0, 0.0],
        "velocity_loop": [10 ** 0.8621, 0.0, 0.0],
        "direction_loop": [10 ** -0.5181, 0.0, 0.0],
        "distance_loop": [10 ** -1.3539, 0.0, 0.0],
        "algorithm": "pid",
        "track": "ABU_F2.csv",
    }
    pid = run_parallel(pid_parameters_tunned, list_in, "pacejka_D")
    plot_line_shadow(list_in, pid, "--", "black", "PID")

    mpc_parameters = {
        "Q": [1, 1, 1, 100],
        "R": [0.1, 0.1],
        "prediction_horizon": 8,
        "algorithm": "mpc",
        "track": "ABU_F2.csv",
    }
    mpc = run_parallel(mpc_parameters, list_in, "pacejka_D")
    plot_line_shadow(list_in, mpc, "--", "red", "MPC")

    N_list = [50, 100, 200, 400]
    for i in range(4):
        deepc_parameters_update = {
            "lambda_g": 5.0 * float(N_list[i]) / 100.0,
            "lambda_y": [2e2] * 4,
            "N": N_list[i],
            "algorithm": "deepc",
        }
        deepc_parameters = mpc_parameters.copy()
        deepc_parameters.update(deepc_parameters_update)
        deepc = run_parallel(deepc_parameters, list_in, "pacejka_D", True)
        plot_line_shadow(
            list_in,
            deepc,
            "-",
            viridis(float(i / len(N_list))),
            "DeePC, N=" + str(N_list[i]),
        )

    plt.xlabel("Track peak attrition (Pacejka D)")
    plt.ylabel("RSS of tracking deviation per step, m")
    ax.set_yscale("symlog", linthresh=1e-1)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.xlim(0.8, 1.6)
    plt.legend()
    plt.grid()
    plt.title("Tires friction variation on Abu Dhabi circuit")
    plt.savefig(f"img/Abu_Dhabi_new.pdf")
    print(f"Look at the picture: img/Abu_Dhabi_new.pdf")


def circle_time():
    """
    Run experiments for circle time tracking and plot results.

    Arguments:
    None

    Returns:
    None
    """
    list_in = range(3000, 4500, 100)
    list_print = [x / 100 for x in list_in]
    # Generating a range of values for circle time tracking.

    # Experiment plotting
    fig, ax = plt.subplots(figsize=(7, 4))
    # Creating subplots for the experiment.

    pid_parameters = {
        "heading_loop": [10 ** 0.389, 0.0, 0.0],
        "velocity_loop": [10 ** 0.8621, 0.0, 0.0],
        "direction_loop": [10 ** -0.5181, 0.0, 0.0],
        "distance_loop": [10 ** -1.3539, 0.0, 0.0],
        "algorithm": "pid",
        "save_folder": "results_circle_time",
    }
    # Setting parameters for PID tracking.

    pid = run_parallel(pid_parameters, list_in, "Lissajous_circle_time")
    # Running parallel experiments for PID tracking.

    plot_line_shadow(list_print, pid, "--", "black", "PID")
    # Plotting PID results.

    mpc_parameters = {
        "Q": [1, 1, 1, 100],
        "R": [0.1, 0.1],
        "prediction_horizon": 8,
        "algorithm": "mpc",
        "save_folder": "results_circle_time",
    }
    # Setting parameters for MPC tracking.

    mpc = run_parallel(mpc_parameters, list_in, "Lissajous_circle_time")
    # Running parallel experiments for MPC tracking.

    plot_line_shadow(list_print, mpc, "--", "red", "MPC")
    # Plotting MPC results.

    N_list = [50, 100, 200, 400]
    # Defining a list of N values for DEEPC tracking.

    for i in range(4):
        deepc_parameters_update = {
            "lambda_g": 5.0 * float(N_list[i]) / 100.0,
            "lambda_y": [2e2] * 4,
            "N": N_list[i],
            "algorithm": "deepc",
        }
        # Updating parameters for DEEPC tracking.

        deepc_parameters = mpc_parameters.copy()
        deepc_parameters.update(deepc_parameters_update)
        # Copying and updating MPC parameters for DEEPC tracking.

        deepc = run_parallel(deepc_parameters, list_in, "Lissajous_circle_time", True)
        # Running parallel experiments for DEEPC tracking.

        plot_line_shadow(
            list_print,
            deepc,
            "-",
            viridis(float(i / len(N_list))),
            "DeePC, N=" + str(N_list[i]),
        )
        # Plotting DEEPC results.

    plt.xlabel("Lissajous total time, s")
    plt.ylabel("RSS of tracking deviation per step, m")
    # Setting labels for x and y axes.

    ax.set_yscale("symlog", linthresh=1e0)
    # Setting y-axis scale.
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.xlim(30, 44)
    plt.legend()
    plt.grid()
    plt.title("Tracking error vs. trace difficulty")
    plt.savefig(f"img/Circle_time_new.pdf")
    print(f"Look at the picture: img/Circle_time_new.pdf")


def ph_vs_ds():
    """
    Run experiments for prediction horizon vs. dataset size tracking and plot results.

    Arguments:
    None

    Returns:
    None
    """
    list_in = [2, 4, 6, 8, 10, 12, 16]
    list_print = [x * 10 for x in list_in]
    # Generating a list of values for prediction horizon.

    # Experiment plotting
    fig, ax = plt.subplots(figsize=(7, 4))
    # Creating subplots for the experiment.

    N_list = [50, 100, 200, 400]
    # Defining a list of N values for DEEPC tracking.

    for i in range(4):
        deepc_parameters = {
            "Q": [1, 1, 1, 100],
            "R": [0.1, 0.1],
            "prediction_horizon": 8,
            "lambda_g": 5.0 * float(N_list[i]) / 100.0,
            "lambda_y": [2e2] * 4,
            "N": N_list[i],
            "algorithm": "deepc",
            "Lissajous_circle_time": 4500,
            "save_folder": "results_ph_ds",
        }
        # Setting parameters for DEEPC tracking.

        deepc = run_parallel(deepc_parameters, list_in, "prediction_horizon", True)
        # Running parallel experiments for DEEPC tracking.

        plot_line_shadow(
            list_print,
            deepc,
            "-",
            viridis(float(i / len(N_list))),
            "DeePC, N=" + str(N_list[i]),
        )
        # Plotting DEEPC results.

    plt.xlabel("Prediction horizon, ms")
    plt.ylabel("RSS of tracking deviation per step, m")
    # Setting labels for x and y axes.

    ax.set_yscale("symlog", linthresh=1e-1)
    # Setting y-axis scale.
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.xlim(20, 160)
    plt.legend()
    plt.grid()
    plt.title("DeePC: Dataset size (N) vs. prediction horizon")
    plt.savefig(f"img/Ds_vs_ph.pdf")
    print(f"Look at the picture: img/Ds_vs_ph.pdf")


def one_step(vec):
    """
    Executes a series of experiments using PID parameters derived from the input vector and calculates a weighted sum of the results.

    Arguments:
    vec : array-like
        A 1D array or list of four numerical values. These values are exponentiated and used as gains for different PID control loops.

    Returns:
    float
        A single numerical value representing the weighted sum of the results from the experiments.
    """
    results = np.zeros(4)  # Initialize an array to store results of the experiments
    weights = np.array([1, 5, 10, 200])  # Weights for each result
    pid_parameters = {
        "heading_loop": [10 ** vec[0], 0.0, 0.0],  # Gain for heading loop
        "velocity_loop": [10 ** vec[1], 0.0, 0.0],  # Gain for velocity loop
        "direction_loop": [10 ** vec[2], 0.0, 0.0],  # Gain for direction loop
        "distance_loop": [10 ** vec[3], 0.0, 0.0],  # Gain for distance loop
        "algorithm": "pid",
        "save_folder": "tunning",
        "Lissajous_circle_time": 4100,  # Initial value for Lissajous circle time
    }
    results[3] = run_experiment(pid_parameters)  # Run experiment and store result

    # Update parameters for further experiments
    pid_parameters.update(
        {
            "Lissajous_circle_time": 5000,  # New value for Lissajous circle time
            "track": "ABU_F2.csv",  # Track file to use
            "pacejka_D": 0.8,  # Pacejka coefficient
        }
    )
    results[0] = run_experiment(pid_parameters)  # Run experiment and store result

    pid_parameters.update({"pacejka_D": 1.2})  # Update Pacejka coefficient
    results[1] = run_experiment(pid_parameters)  # Run experiment and store result

    pid_parameters.update({"pacejka_D": 1.6})  # Update Pacejka coefficient
    results[2] = run_experiment(pid_parameters)  # Run experiment and store result

    return np.sum(results * weights)  # Calculate and return the weighted sum of results


def pid_tunning():
    """
    Tunes PID parameters by running multiple experiments in parallel and finds the parameter set that produces the minimum result.

    No arguments.

    Prints:
    The best parameter set (as a list) and the corresponding minimal result.
    """
    np.random.seed(1)  # Set random seed for reproducibility
    cpu_n = 96 * 50  # Number of parameter sets to generate
    param_list = [
        np.random.uniform(-2, 1, 4) for _ in range(cpu_n)
    ]  # Generate random parameter sets

    pool = multiprocessing.Pool()  # Create a multiprocessing pool
    execution_result = pool.map(
        one_step, param_list
    )  # Map 'one_step' function to each parameter set
    pool.close()  # Close the pool
    pool.join()  # Wait for all processes to complete

    index = np.argmin(execution_result)  # Find the index of the minimum result
    print(param_list[index])  # Print the best parameter set
    print(execution_result[index])  # Print the minimal result


if __name__ == "__main__":        
    # Execute the following functions when the script is run as the main program.

    ph_vs_ds()
    # Run the experiment for prediction horizon vs. dataset size tracking.
    
    pid_tunning()
    # Run the algorithm for setting PID parameters

    circle_time()
    # Run the experiment for circle time tracking.

    abu_dhabi()
    # Display information or perform actions related to Abu Dhabi.

    nice_abu_dhabi_picture()
    # Display a nice picture of Abu Dhabi.
