import pytest
import sys
sys.path.insert(1, 'source/')
import model 

def test_always_passes():
    assert True

def test_one_handred_acceleration_dyn():
    obj = model.Dynamic_model()
    act = model.Racecar_Action(1.0, 0.0)
    res = model.Racecar_State()
    time = 0.0
    while res.speed < 100.0/3.6:
        res = obj.Step(act)
        time += 0.01
    assert time < 3.0
    
def test_two_handred_acceleration_dyn():
    obj = model.Dynamic_model()
    act = model.Racecar_Action(1.0, 0.0)
    res = model.Racecar_State()
    time = 0.0
    while res.speed < 200.0/3.6:
        res = obj.Step(act)
        time += 0.01
    assert time < 7.0
    
def test_max_speed():
    obj = model.Dynamic_model()
    act = model.Racecar_Action(1.0, 0.0)
    res = model.Racecar_State()
    res_previous = model.Racecar_State()
    init = True
    while res.speed > res_previous.speed or init:
        res_previous = res
        init = False
        res = obj.Step(act)
    assert res.speed*3.6 > 290.0
    
def test_one_handred_acceleration_kin():
    obj = model.Kinematic_model()
    act = model.Racecar_Action(1.0, 0.0)
    res = model.Racecar_State()
    time = 0.0
    while res.speed < 100.0/3.6:
        res = obj.Step(act)
        time += 0.01
    assert time < 3.0
    
    
def test_two_handred_acceleration_kin():   
    obj = model.Kinematic_model()
    act = model.Racecar_Action(1.0, 0.0)
    res = model.Racecar_State()
    time = 0.0
    while res.speed < 200.0/3.6:
        res = obj.Step(act)
        time += 0.01
    assert time < 7.0
    