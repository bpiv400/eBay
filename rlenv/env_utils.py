"""
Utility functions for use in objects related to the RL environment
"""
import utils
from constants import SLR_PREFIX, BYR_PREFIX
from rlenv.env_consts import MODEL_DIR


def model_str(model_name, byr=False):
    """
    returns the string giving the name of an offer model
    model (used to refer to the model in SimulatorInterface
     and Composer

    :param model_name: str giving base name
    :param byr: boolean indicating whether this is a buyer model
    :return:
    """
    if not byr:
        name = '{}_{}'.format(SLR_PREFIX, model_name)
    else:
        name = '{}_{}'.format(BYR_PREFIX, model_name)
    return name


def load_featnames(model_type, model_name):
    dir_path = '{}/{}/{}/'.format(MODEL_DIR, model_type, model_name)
    featnames_path = '{}featnames.pkl'.format(dir_path)
    featnames_dict = utils.unpickle(featnames_path)
    return featnames_dict
