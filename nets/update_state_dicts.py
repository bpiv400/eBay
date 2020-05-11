"""
Updates the model state dictionaries from the old structure where
there the output layer was the last layer of a set of sequential layers
to the new structure where it's called output
"""
import argparse
import os
import torch
from constants import MODELS, PREFIX, MODEL_DIR
from utils import load_state_dict, load_sizes
from nets.nets_consts import LAYERS_FULL
from nets.FeedForward import FeedForward
BACKUP_DIR = '{}/backup/'.format(PREFIX)


def update_dicts():
    if not os.path.isdir(BACKUP_DIR):
        os.mkdir(BACKUP_DIR)
    for model in MODELS:
        state_dict = load_state_dict(model)
        backup_path = get_backup_path(model)
        torch.save(state_dict, backup_path)
        fully_connected_compat(state_dict=state_dict)
        check_compat(model_name=model, state_dict=state_dict)


def get_backup_path(model):
    return '{}{}.net'.format(BACKUP_DIR, model)


def check_compat(model_name=None, state_dict=None):
    sizes = load_sizes(model_name)
    net = FeedForward(sizes)
    try:
        net.load_state_dict(state_dict, strict=True)
        print('{} is compatible after update'.format(model_name))
    except RuntimeError as err:
        print('{} is not compatible after update'.format(model_name))
        print(err)
        exit(1)


def fully_connected_compat(state_dict=None):
    """
    Renames the parameters in a torch state dict generated
    when output layer lived in FullyConnected to be compatible
    with separate output layer
    :param state_dict: dictionary name -> param
    :return: dict
    """
    old_prefix = 'nn1.seq.{}'.format(LAYERS_FULL)
    new_prefix = 'output'
    substitute_prefix(old_prefix=old_prefix, new_prefix=new_prefix,
                      state_dict=state_dict)


def substitute_prefix(old_prefix=None, new_prefix=None, state_dict=None):
    effected_keys = list()
    for key in state_dict.keys():
        if key[:len(old_prefix)] == old_prefix:
            effected_keys.append(key)
    # null case
    if len(effected_keys) == 0:
        return state_dict

    # replace each old prefix with a new prefix
    for effected_key in effected_keys:
        effected_suffix = effected_key[len(old_prefix):]
        new_key = '{}{}'.format(new_prefix, effected_suffix)
        state_dict[new_key] = state_dict[effected_key]
        del state_dict[effected_key]


def replace_dicts():
    for model in MODELS:
        curr_path = get_backup_path(model)
        state_dict = torch.load(curr_path, map_location=torch.device('cpu'))
        standard_path = '{}{}.net'.format(MODEL_DIR, model)
        fully_connected_compat(state_dict=state_dict)
        check_compat(model_name=model, state_dict=state_dict)
        torch.save(state_dict, standard_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=1)
    step = parser.parse_args().step
    if step == 1:
        update_dicts()
        print('Done storing backups')
    else:
        replace_dicts()
        print('Done replacement')


if __name__ == '__main__':
    main()
