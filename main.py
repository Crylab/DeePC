#!/usr/bin/env python
import sys
sys.path.insert(1, 'source/')
import deepc_tracking
import pid_tracking
import mpc_tracking
import graph
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib import cm

viridis = cm.get_cmap('viridis', 12)

def run_experiment(dict_in: dict):
    if dict_in["algorithm"] == "pid":
        obj = pid_tracking.PID_Tracking(dict_in)
    elif dict_in["algorithm"] == "deepc":
        obj = deepc_tracking.DEEPC_Tracking(dict_in)
    elif dict_in["algorithm"] == "mpc":
        obj = mpc_tracking.MPC_Tracking(dict_in)
    result = obj.trajectory_tracking()
    rss = obj.rss
    return rss

def run_parallel(dict_in: dict, list_in: list, var_param: str, seeds: bool = False):
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
    
    # Execution
    pool = multiprocessing.Pool()
    execution_result = pool.map(run_experiment, param_list)
    pool.close()
    pool.join()
    
    average_list = []
    sigma_list = []
    counter = 0
    for _ in range(len(list_in)):
        if seeds:
            seed_list = []
            for seed in range(30):
                seed_list.append(execution_result[counter])
                counter +=1
            average = sum(seed_list)/len(seed_list)
            average_list.append(average)
            seed_average_nd = np.array(seed_list)
            sigma = np.sqrt(sum(np.square(seed_average_nd-average))/seed_average_nd.size)
            sigma_list.append(sigma)
        else:
            average_list.append(execution_result[counter])
            counter +=1
            sigma_list.append(0)
        
            
    return average_list, sigma_list

def plot_line_shadow(list_in: list, res: tuple, linetype: str, color, name: str):
    plt.plot(list_in, res[0], label=name, color=color, linestyle = linetype)
    plt.fill_between(
        list_in, 
        [res[0][ogni] - res[1][ogni] for ogni in range(len(list_in))], 
        [res[0][ogni] + res[1][ogni] for ogni in range(len(list_in))], 
        alpha=0.3, 
        color=color
    )

def nice_abu_dhabi_picture():
    # PID part
    parameters = {
        "heading_loop": [1.1, 0.0, 0.0],
        "velocity_loop": [3.5, 0.0, 0.0],
        "direction_loop": [0.7, 0.0, 0.0],
        "distance_loop": [0.32, 0.0, 0.0],
        "track": "ABU_F2.csv",
        "pacejka_D": 2.0,
    }
    obj = pid_tracking.PID_Tracking(parameters)
    pid_result = obj.trajectory_tracking()
    
    # DEEPC part
    parameters = {
        'lambda_g': 5.0,
        'lambda_y': [2e2] * 4,
        'N': 100,
        'track': "ABU_F2.csv",
        'Q': [1, 1, 1, 100],
        'R': [0.1, 0.1],
        'prediction_horizon': 8,
        'track_upper_bound': 100.0,
        "pacejka_D": 2.0,
    }
    obj = deepc_tracking.DEEPC_Tracking(parameters)
    deepc_result = obj.trajectory_tracking()
    
    # MPC part
    parameters = {
        'track': "ABU_F2.csv",
        'Q': [1, 1, 1, 100],
        'R': [0.1, 0.1],
        'prediction_horizon': 8,
        'track_upper_bound': 100.0,
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
    visual.add_state_landscape(obj.trajectory, '--')
    visual.add_state_path(mpc_result, "red", 250, "MPC")
    visual.add_state_path(pid_result, "black", 500, "PID")
    visual.compression(10)
    visual.generate_pic_at('img/Nice_abu_dhabi.pdf', 80.0)
 
def abu_dhabi():
    
    list_in = np.arange(0.8, 1.6, 0.01)
    # Experiment plotting
    fig, ax = plt.subplots()
    
    pid_parameters = {
        "heading_loop": [2.0, 0.0, 0.0],
        "velocity_loop": [3.0, 0.0, 0.0],
        "direction_loop": [0.1, 0.0, 0.0],
        "distance_loop": [0.3, 0.0, 0.0],
        "algorithm": "pid",
        'track': "ABU_F2.csv",
    }
    pid = run_parallel(pid_parameters, list_in, "pacejka_D")
    plot_line_shadow(list_in, pid, '--', "black", "PID")
    
    mpc_parameters = {
        'Q': [1, 1, 1, 100],
        'R': [0.1, 0.1],
        'prediction_horizon': 8,
        "algorithm": "mpc",
        'track': "ABU_F2.csv",
    }
    mpc = run_parallel(mpc_parameters, list_in, "pacejka_D")
    plot_line_shadow(list_in, mpc, '--', "red", "MPC")
    
    N_list = [50, 100, 200, 400]
    for i in range(4):
        deepc_parameters_update = {
            'lambda_g': 5.0*float(N_list[i])/100.0,
            'lambda_y': [2e2] * 4,
            'N': N_list[i],
            "algorithm": "deepc"
        }
        deepc_parameters = mpc_parameters.copy()
        deepc_parameters.update(deepc_parameters_update)
        deepc = run_parallel(deepc_parameters, list_in, "pacejka_D", True)
        plot_line_shadow(list_in, deepc, '-', viridis(float(i/len(N_list))), "DeePC, N="+str(N_list[i]))
    
    plt.xlabel("Track peak attrition (Pacejka D)")
    plt.ylabel("RSS of tracking deviation per step, m")
    #ax.set_yscale('log')
    ax.set_yscale('symlog', linthresh=1e-1)
    #ax.set_xscale('log')
    plt.legend()
    plt.grid()
    plt.title("Tires friction variation on Abu Dhabi circuit")
    plt.savefig(f"img/Abu_Dhabi_new.pdf")
    print(f"Look at the picture: img/Abu_Dhabi_new.pdf")
       
def circle_time():
    
    list_in = range(3000, 4500, 100)
    # Experiment plotting
    fig, ax = plt.subplots()
    
    pid_parameters = {
        "heading_loop": [2.0, 0.0, 0.0],
        "velocity_loop": [3.0, 0.0, 0.0],
        "direction_loop": [0.1, 0.0, 0.0],
        "distance_loop": [0.3, 0.0, 0.0],
        "algorithm": "pid"
    }
    pid = run_parallel(pid_parameters, list_in, "Lissajous_circle_time")
    plot_line_shadow(list_in, pid, '--', "black", "PID")
    
    mpc_parameters = {
        'Q': [1, 1, 1, 100],
        'R': [0.1, 0.1],
        'prediction_horizon': 8,
        "algorithm": "mpc"
    }
    mpc = run_parallel(mpc_parameters, list_in, "Lissajous_circle_time")
    plot_line_shadow(list_in, mpc, '--', "red", "MPC")
    
    N_list = [50, 100, 200, 400]
    for i in range(4):
        deepc_parameters_update = {
            'lambda_g': 5.0*float(N_list[i])/100.0,
            'lambda_y': [2e2] * 4,
            'N': N_list[i],
            "algorithm": "deepc"
        }
        deepc_parameters = mpc_parameters.copy()
        deepc_parameters.update(deepc_parameters_update)
        deepc = run_parallel(deepc_parameters, list_in, "Lissajous_circle_time", True)
        plot_line_shadow(list_in, deepc, '-', viridis(float(i/len(N_list))), "DeePC, N="+str(N_list[i]))
    
    plt.xlabel("Lissajous total time, s")
    plt.ylabel("RSS of tracking deviation per step, m")
    #ax.set_yscale('log')
    ax.set_yscale('symlog', linthresh=1e0)
    #ax.set_xscale('log')
    plt.legend()
    plt.grid()
    plt.title("Tracking error vs. trace difficulty")
    plt.savefig(f"img/Circle_time_new.pdf")
    print(f"Look at the picture: img/Circle_time_new.pdf")
    
def pid_optimization_abu_dhabi():
    list_in = np.arange(0.8, 1.6, 0.01)
    def cost_function(vec: list):
        parameters = {
            "heading_loop": [vec[0], 0.0, 0.0],
            "velocity_loop": [vec[1], 0.0, 0.0],
            "direction_loop": [vec[2], 0.0, 0.0],
            "distance_loop": [vec[3], 0.0, 0.0],
            'track': "ABU_F2.csv",
        }
        res, _ = run_parallel(parameters, list_in, "pacejka_D")
        return sum(res)
    
    def gradient(vec: list):
        base = cost_function(vec)
        print("Base cost: "+str(base))
        step = 0.01
        result = []
        for i in range(4):
            local_vec = vec.copy()
            local_vec[i] += step
            local = cost_function(local_vec)
            result.append((local-base)/step)
        return result

    alpha = 0.1
    for iter in range(100):
        vec = [1.0, 1.0, 0.1, 0.1]
        g_vec = gradient(vec)
        for i in range(4):
            vec[i] -= g_vec[i] * alpha
    print("Resultance vec" + str(vec))
    return vec
    
def ph_vs_ds():
    list_in = [2, 4, 6, 8, 10, 12, 16]
    # Experiment plotting
    fig, ax = plt.subplots()
    N_list = [50, 100, 200, 400]
    for i in range(4):
        deepc_parameters = {
            'Q': [1, 1, 1, 100],
            'R': [0.1, 0.1],
            'prediction_horizon': 8,
            'lambda_g': 5.0*float(N_list[i])/100.0,
            'lambda_y': [2e2] * 4,
            'N': N_list[i],
            "algorithm": "deepc",
            "Lissajous_circle_time": 4500,
        }
        deepc = run_parallel(deepc_parameters, list_in, "prediction_horizon", True)
        plot_line_shadow(list_in, deepc, '-', viridis(float(i/len(N_list))), "DeePC, N="+str(N_list[i]))
    plt.xlabel("Lissajous total time, s")
    plt.ylabel("RSS of tracking deviation per step, m")
    #ax.set_yscale('log')
    ax.set_yscale('symlog', linthresh=1e0)
    #ax.set_xscale('log')
    plt.legend()
    plt.grid()
    plt.title("Tracking error vs. trace difficulty")
    plt.savefig(f"img/Circle_time_new.pdf")
    print(f"Look at the picture: img/Circle_time_new.pdf")
    
if __name__ == "__main__":
    
    pid_optimization_abu_dhabi()
    
    #circle_time()
    #abu_dhabi()
    exit()
    print('Hello world!')
    parameters = {
        "heading_loop": [1.1, 0.0, 0.0],
        "velocity_loop": [3.5, 0.0, 0.0],
        "direction_loop": [0.7, 0.0, 0.0],
        "distance_loop": [0.32, 0.0, 0.0],
        "track": "ABU_F2.csv",
        "pacejka_D": 2.0,
    }
    obj = pid_tracking.PID_Tracking(parameters)
    result = obj.trajectory_tracking()
    
    list_in = range(3000, 4500, 100)
    
    pid_res = pid_parallel_run_lis(list_in, "Lissajous_circle_time")
    pid_res_loose = pid_parallel_run_loose_lis(list_in, "Lissajous_circle_time")
    mpc_res = mpc_parallel_run_lis(list_in, "Lissajous_circle_time")
    deepc_res50, deepc_sig_50 = deepc_parallel_run_lis(list_in, "Lissajous_circle_time", 50)
    deepc_res100, deepc_sig_100 = deepc_parallel_run_lis(list_in, "Lissajous_circle_time", 100)
    deepc_res200, deepc_sig_200 = deepc_parallel_run_lis(list_in, "Lissajous_circle_time", 200)
    deepc_res400, deepc_sig_400 = deepc_parallel_run_lis(list_in, "Lissajous_circle_time", 400)
    
    # Experiment plotting
    fig, ax = plt.subplots()
    plt.plot(list_in, pid_res, label="PID agressive", linestyle='--', color='black')
    plt.plot(list_in, pid_res_loose, label="PID loose", linestyle='-.', color='black')
    plt.plot(list_in, mpc_res, label="MPC", linestyle='--', color='red')
    plt.plot(list_in, deepc_res50, label="DeePC, N=50", color=viridis(float(0.25)))
    plt.fill_between(
        list_in, 
        [deepc_res50[ogni] - deepc_sig_50[ogni] for ogni in range(len(list_in))], 
        [deepc_res50[ogni] + deepc_sig_50[ogni] for ogni in range(len(list_in))], 
        alpha=0.3, 
        color=viridis(0.25)
    )
    plt.plot(list_in, deepc_res100, label="DeePC, N=100", color=viridis(float(0.5)))
    plt.fill_between(
        list_in, 
        [deepc_res100[ogni] - deepc_sig_100[ogni] for ogni in range(len(list_in))], 
        [deepc_res100[ogni] + deepc_sig_100[ogni] for ogni in range(len(list_in))], 
        alpha=0.3, 
        color=viridis(0.5)
    )
    plt.plot(list_in, deepc_res200, label="DeePC, N=200", color=viridis(float(0.75)))
    plt.fill_between(
        list_in, 
        [deepc_res200[ogni] - deepc_sig_200[ogni] for ogni in range(len(list_in))], 
        [deepc_res200[ogni] + deepc_sig_200[ogni] for ogni in range(len(list_in))], 
        alpha=0.3, 
        color=viridis(0.75)
    )
    plt.plot(list_in, deepc_res400, label="DeePC, N=400", color=viridis(float(1.0)))
    plt.fill_between(
        list_in, 
        [deepc_res400[ogni] - deepc_sig_400[ogni] for ogni in range(len(list_in))], 
        [deepc_res400[ogni] + deepc_sig_400[ogni] for ogni in range(len(list_in))], 
        alpha=0.3, 
        color=viridis(1.0)
    )
    plt.xlabel("Lissajous total time, s")
    plt.ylabel("RSS of tracking deviation per step, m")
    #ax.set_yscale('log')
    ax.set_yscale('symlog', linthresh=1e0)
    #ax.set_xscale('log')
    plt.legend()
    plt.grid()
    plt.title("Tracking error vs. trace difficulty")
    plt.savefig(f"img/Circle_time.pdf")
    print(f"Look at the picture: img/Circle_time.pdf")
    
    