import pytest
import sys
sys.path.insert(1, 'source/')
import pid_tracking 
import deepc_tracking
import mpc_tracking

def test_always_passes():
    assert True

def test_pid_tracking():
    obj = pid_tracking.PID_Tracking()
    obj.trajectory_tracking()
    rss = obj.get_rss()
    assert rss < 2.0
    
def test_deepc_tracking():
    parameters = {
        "Q": [1.0, 1.0, 10.0, 10.0],
        "R": [0.1, 0.1],
        "lambda_y": [50]*4,
        "lambda_g": 1.0,
        "prediction_horizon": 8,
    }
    obj = deepc_tracking.DEEPC_Tracking(parameters)
    obj.trajectory_tracking()
    rss = obj.get_rss()
    assert rss < 0.2
    
def test_mpc_tracking():
    parameters = {
        "Q": [1.0, 1.0, 10.0, 10.0],
        "R": [0.1, 0.1],
        "prediction_horizon": 8,
    }
    obj = mpc_tracking.MPC_Tracking(parameters)
    obj.trajectory_tracking()
    rss = obj.get_rss()
    assert rss < 1.0