import argparse
import pandas as pd
from compress_pickle import load, dump
from constants import INPUT_DIR, INDEX_DIR, PARTS_DIR, MODELS,\
    VALIDATION, PARTITIONS, POLICY_MODELS, CLEAN_DIR
from rlenv.util import (get_env_sim_subdir, load_featnames,
                        get_env_sim_dir, load_chunk)
from utils import subset_lstgs


def load_all_inputs(part=None, lstgs=None):
    input_dir = '{}{}/'.format(INPUT_DIR, part)
    index_dir = '{}{}/'.format(INDEX_DIR, part)
    inputs_dict = dict()
    for model in MODELS + POLICY_MODELS:
        inputs_dict[model] = load_model_inputs(model=model,
                                               input_dir=input_dir,
                                               index_dir=index_dir,
                                               lstgs=lstgs)
    return inputs_dict


def load_model_inputs(model=None, input_dir=None, index_dir=None, lstgs=None):
    """
    For the given model, generates a dictionary where the keys give
    feature sets for the model, and the values give pandas dataframes
    containing the input data for those feature sets
    """
    # load input and index files and featnames
    input_path = '{}{}.gz'.format(input_dir, model)
    index_path = '{}{}.gz'.format(index_dir, model)
    index = load(index_path)
    featnames = load_featnames(model)
    inputs = load(input_path)['x']
    contains = None
    # iterate over feature sets
    for feat_set_name in inputs.keys():
        cols = featnames['offer'] if 'offer' in feat_set_name else featnames[feat_set_name]
        # contstruct a dataframe containing the inputs for the current feature set
        inputs_df = pd.DataFrame(data=inputs[feat_set_name],
                                 index=index,
                                 columns=cols)
        # on the first feature set, generate a boolean series that indicates whether each
        # lstg is contained in the current chunk
        if contains is None:
            full_lstgs = inputs_df.index.get_level_values('lstg')
            contains = full_lstgs.isin(lstgs)
        # subset the input dataframe to only contain lstgs in the chunk
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


def lstgs_without_duplicated_timestamps(lstgs=None):
    # load timestamps
    offers = load(CLEAN_DIR + 'offers.pkl').reindex(index=lstgs, level='lstg')

    # remove censored offers
    clock = offers.loc[~offers.censored, 'clock']

    # remove duplicate timestamps within thread
    toDrop = clock.groupby(['lstg', 'thread']).apply(lambda x: x.duplicated())
    clock = clock[~toDrop]

    # flag listings with duplicate timestamps across threads
    flag = clock.groupby('lstg').apply(lambda x: x.duplicated())
    flag = flag.groupby('lstg').max()

    # drop flagged listgins
    lstgs = lstgs.drop(flag[flag].index)
    return lstgs


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', required=True, type=int, help='chunk number',
                        default=1)
    parser.add_argument('--part', required=True, type=str, help='partition name',
                        default=VALIDATION, choices=PARTITIONS)
    args = parser.parse_args()
    part, num = args.part, args.num

    # path
    base_dir = get_env_sim_dir(part)

    # index of listings
    print('Loading chunk...')
    _, lookup = load_chunk(base_dir=base_dir, num=num)
    lstgs = lookup.sort_index().index
    lstgs = lstgs_without_duplicated_timestamps(lstgs=lstgs)

    # model inputs
    print('Loading model inputs...')
    model_inputs = load_all_inputs(part=part, lstgs=lstgs)

    # other components
    print('Loading x offer and x_thread...')
    x_thread, x_offer = load_outcomes(part=part, lstgs=lstgs)

    # save output
    output = {
        'inputs': model_inputs,
        'x_thread': x_thread,
        'x_offer': x_offer
    }
    print('Dumping file...')
    subdir = get_env_sim_subdir(base_dir=base_dir, chunks=True)
    path = '{}{}_test.gz'.format(subdir, num)
    dump(output, path)


if __name__ == '__main__':
    main()
