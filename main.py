#!/usr/bin/env python
import sys
sys.path.insert(1, 'source/')
import deepc_tracking
import pid_tracking
import mpc_tracking
import graph

if __name__ == "__main__":
    print('Hello world!')
    parameters = {
        "Q": [1.0, 1.0, 10.0, 10.0],
        "R": [0.1, 0.1],
        "print_out": "Calculations",
        "prediction_horizon": 8,
        "track_upper_bound": 100.0,
    }
    obj = mpc_tracking.MPC_Tracking(parameters)
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
    