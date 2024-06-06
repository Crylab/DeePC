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

def pid_runnung(dict_in: dict):
    
    obj = pid_tracking.PID_Tracking(dict_in)
    result = obj.trajectory_tracking()
    rss = obj.rss
    return rss

def deepc_runnung(dict_in: dict):
    
    obj = deepc_tracking.DEEPC_Tracking(dict_in)
    result = obj.trajectory_tracking()
    rss = obj.rss
    return rss

def mpc_running(dict_in: dict):
    obj = deepc_tracking.MPC_Tracking(dict_in)
    result = obj.trajectory_tracking()
    rss = obj.rss
    return rss

def mpc_parallel_run(list_in: list, var_param: str):
    parameters = {
        'track': "ABU_F2.csv",
        'Q': [1, 1, 1, 100],
        'R': [0.1, 0.1],
        'prediction_horizon': 8,
        'track_upper_bound': 100.0,
    }
    param_list = []
    for pac in list_in:
        local_param = parameters.copy()
        local_param[var_param] = pac
        param_list.append(local_param)
        
    # Execution
    pool = multiprocessing.Pool()
    execution_result = pool.map(deepc_runnung, param_list)
    pool.close()
    pool.join()
        
    return execution_result

def pid_parallel_run(list_in: list, var_param: str):
    parameters = {
        "heading_loop": [2.0, 0.0, 0.0],
        "velocity_loop": [3.0, 0.0, 0.0],
        "direction_loop": [0.1, 0.0, 0.0],
        "distance_loop": [0.3, 0.0, 0.0],
        "track": "ABU_F2.csv",
        "pacejka_D": 1.0,
        'track_upper_bound': 100.0,
    }
    param_list = []
    for pac in list_in:
        local_param = parameters.copy()
        local_param[var_param] = pac
        param_list.append(local_param)
        
    # Execution
    pool = multiprocessing.Pool()
    execution_result = pool.map(pid_runnung, param_list)
    pool.close()
    pool.join()
        
    return execution_result

def pid_parallel_run_loose(list_in: list, var_param: str):
    parameters = {
        "heading_loop": [2.0, 0.0, 0.0],
        "velocity_loop": [3.0, 0.0, 0.0],
        "direction_loop": [0.1, 0.0, 0.0],
        "distance_loop": [0.1, 0.0, 0.0],
        "track": "ABU_F2.csv",
        "pacejka_D": 1.0,
        'track_upper_bound': 100.0,
    }
    param_list = []
    for pac in list_in:
        local_param = parameters.copy()
        local_param[var_param] = pac
        param_list.append(local_param)
        
    # Execution
    pool = multiprocessing.Pool()
    execution_result = pool.map(pid_runnung, param_list)
    pool.close()
    pool.join()
        
    return execution_result

def deepc_parallel_run(list_in: list, var_param: str, N:int):
    parameters = {
        'lambda_g': 5.0*float(N)/100.0,
        'lambda_y': [2e2] * 4,
        'N': N,
        'track': "ABU_F2.csv",
        'Q': [1, 1, 1, 100],
        'R': [0.1, 0.1],
        'prediction_horizon': 8,
        'track_upper_bound': 100.0,
    }
    param_list = []
    for pac in list_in:
        local_param = parameters.copy()
        local_param[var_param] = pac
        for seed in range(30):
            local_local_param = local_param.copy()
            local_local_param["seed"] = seed
            param_list.append(local_local_param)
    
    # Execution
    pool = multiprocessing.Pool()
    execution_result = pool.map(deepc_runnung, param_list)
    pool.close()
    pool.join()
    
    average_list = []
    sigma_list = []
    counter = 0
    for _ in range(len(list_in)):
        seed_list = []
        for seed in range(30):
            seed_list.append(execution_result[counter])
            counter +=1
        average = sum(seed_list)/len(seed_list)
        average_list.append(average)
        seed_average_nd = np.array(seed_list)
        sigma = np.sqrt(sum(np.square(seed_average_nd-average))/seed_average_nd.size)
        sigma_list.append(sigma)
        
    return average_list, sigma_list
    
def abu_dhabi_chart():
     # Execution of the experiments
    list_in = np.arange(1.0, 2.0, 0.1)
    pid_res = pid_parallel_run(list_in, "pacejka_D")
    pid_res_loose = pid_parallel_run_loose(list_in, "pacejka_D")
    deepc_res50, deepc_sig_50 = deepc_parallel_run(list_in, "pacejka_D", 50)
    deepc_res100, deepc_sig_100 = deepc_parallel_run(list_in, "pacejka_D", 100)
    deepc_res200, deepc_sig_200 = deepc_parallel_run(list_in, "pacejka_D", 200)
    deepc_res400, deepc_sig_400 = deepc_parallel_run(list_in, "pacejka_D", 400)
    mpc_res = mpc_parallel_run(list_in, "pacejka_D")
    
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
    plt.xlabel("Track peak attrition (Pacejka D)")
    plt.ylabel("RSS of tracking deviation per step, m")
    #ax.set_yscale('log')
    ax.set_yscale('symlog', linthresh=1e-1)
    #ax.set_xscale('log')
    plt.legend()
    plt.grid()
    plt.title("Tires friction variation on Abu Dhabi circuit")
    plt.savefig(f"img/Abu_Dhabi.pdf")
    print(f"Look at the picture: img/Abu_Dhabi.pdf")
    
if __name__ == "__main__":
    
    abu_dhabi_chart()
    exit()
    print('Hello world!')
    parameters = {
        "heading_loop": [1.1, 0.0, 0.0],
        "velocity_loop": [3.5, 0.0, 0.0],
        "direction_loop": [0.7, 0.0, 0.0],
        "distance_loop": [0.32, 0.0, 0.0],
        "track": "ABU_F2.csv",
        "pacejka_D": 1.0,
    }
    obj = pid_tracking.PID_Tracking(parameters)
    result = obj.trajectory_tracking()
    rss = obj.get_rss()    
    print(rss)
    
    visual_params = {
        "name": "PID Tracking",
        "xmin": -100,
        "ymin": -100,
        "xmax": 100,
        "ymax": 100,
        "vehicle_length": 5,
        "vehicle_width": 2,
    }
    
    visual = graph.graph(visual_params)
    visual.add_state_path(result)
    visual.compression(10)
    
    #visual.generate_gif('img/mpc.gif')
    