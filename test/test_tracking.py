import pytest
import sys
sys.path.insert(1, 'source/')
import pid_tracking 

def test_always_passes():
    assert True

def test_pid_tracking():
    obj = pid_tracking.PID_Tracking()
    obj.trajectory_tracking()
    rss = obj.get_rss()
    assert rss < 2.0