"""
Misc. utility functions for RL training

Functions:
    unpickle: wrapper function for pickling an arbitrary object
    actor_map: returns dictionary mapping names of actor classes to classes
"""
import pickle
from deprecated.actor import DeterministicActor
from models import Simulator


def is_none(obj, name='All arguments and keyword arguments'):
    '''
    Function that checks whether a given
    argument is None and throws an error if so

    Args:
        obj: arbitrary object that should not be None

    Kwargs:
        name: name of the object in the calling function
        (Default: 'All arguments and keyword arguments')
    '''
    if obj is None:
        raise ValueError("%s must be defined" % name)


def extract_datatype(name):
    '''
    Extracts dataset type (toy, train, test) from chunk name
    of the form type-num (e.g. toy-1)
    '''
    datatype, _ = name.split('-')
    return datatype


def simulator_map():
    '''
    Defines dictionary mapping strings of simulator class names
    to corresponding classes
    '''
    sim_map = {}
    sim_map['model'] = Simulator
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
