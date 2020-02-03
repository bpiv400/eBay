import argparse
import pandas as pd
from compress_pickle import load, dump
from constants import INPUT_DIR, INDEX_DIR, PARTS_DIR
from rlenv.env_consts import MODELS
from rlenv.env_utils import (get_env_sim_subdir, load_featnames,
                             get_env_sim_dir, load_chunk)


def load_all_inputs(part=None, lstgs=None):
    input_dir = '{}{}/'.format(INPUT_DIR, part)
    index_dir = '{}{}/'.format(INDEX_DIR, part)
    inputs_dict = dict()
    for model in MODELS:
        inputs_dict[model] = load_model_inputs(model=model,
                                               input_dir=input_dir,
                                               index_dir=index_dir,
                                               lstgs=lstgs)
    return inputs_dict


def load_model_inputs(model=None, input_dir=None, index_dir=None, lstgs=None):
    input_path = '{}{}'.format(input_dir, model)
    index_path = '{}{}'.format(index_dir, model)
    index = load(index_path)
    featnames = load_featnames(model)
    inputs = load(input_path)['x']
    contains = None
    for feat_set_name in list(inputs.keys()):
        inputs_df = pd.DataFrame(data=inputs[feat_set_name],
                                 index=index,
                                 columns=featnames[feat_set_name])
        if contains is None:
            full_lstgs = inputs_df.index.get_level_values('lstg')
            contains = full_lstgs.isin(lstgs)
        inputs_df = inputs_df.loc[contains, :]
        inputs[feat_set_name] = inputs_df
    return inputs


def load_outcomes(part=None, lstgs=None):
    outcome_dir = '{}{}/'.format(PARTS_DIR, part)
    x_thread = load('{}x_thread.gz'.format(outcome_dir))
    x_offer = load('{}x_offer.gz'.format(outcome_dir))
    x_thread = subset_lstgs(df=x_thread, lstgs=lstgs)
    x_offer = subset_lstgs(df=x_offer, lstgs=lstgs)
    x_offer['censored'] = x_offer.exp & (x_offer.delay < 1)
    return x_thread, x_offer


def subset_lstgs(df=None, lstgs=None):
    full_lstgs = df.index.get_level_values('lstg')
    contains = full_lstgs.isin(lstgs)
    df = df.loc[contains, :]
    return df


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', required=True, type=int, help='chunk number')
    parser.add_argument('--part', required=True, type=str, help='partition name')
    args = parser.parse_args()
    base_dir = get_env_sim_dir(args.part)
    print('Loading chunk...')
    chunk = load_chunk(base_dir=base_dir, num=args.num)
    lstgs = chunk['lookup'].index
    print('Loading model inputs...')
    model_inputs = load_all_inputs(part=args.part, lstgs=lstgs)
    print('Loading x offer and x_thread...')
    x_thread, x_offer = load_outcomes(part=args.part, lstgs=lstgs)
    output = {
        'inputs': model_inputs,
        'x_thread': x_thread,
        'x_offer': x_offer
    }
    subdir = get_env_sim_subdir(base_dir=base_dir, chunks=True)
    path = '{}1_test.gz'.format(subdir)
    dump(output, path)


if __name__ == '__main__':
    main()
