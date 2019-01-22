"""
Misc. utility functions for RL training

Functions:
    unpickle: wrapper function for pickling an arbitrary object
    actor_map: returns dictionary mapping names of actor classes to classes
"""
import re
import pickle
from actor import DeterministicActor
from models import Model


def simulator_map():
    '''
    Defines dictionary mapping strings of simulator class names
    to corresponding classes
    '''
    sim_map = {}
    sim_map['model'] = Model
    return sim_map


def actor_map():
    '''
    Defines dictionary mapping strings of actor class names
    to corresponding classes
    '''
    act_map = {}
    act_map['DeterministicActor'] = DeterministicActor
    return act_map


def unpickle(path):
    '''
    Extracts an abritrary object from the pickle located at path and returns
    that object

    Args:
        path: string denoting path to pickle

    Returns:
        arbitrary object contained in pickle
    '''
    f = open(path, "rb")
    obj = pickle.load(f)
    f.close()
    return obj
