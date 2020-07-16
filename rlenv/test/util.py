from compress_pickle import load
import pandas as pd
from rlenv.util import load_featnames
from utils import load_file
from constants import MODELS, INPUT_DIR, INDEX_DIR
from featnames import CENSORED, EXP, DELAY


def load_model_inputs(model=None, input_dir=None, index_dir=None, lstgs=None):
    input_path = '{}{}.gz'.format(input_dir, model)
    index_path = '{}{}.gz'.format(index_dir, model)
    index = load(index_path)
    featnames = load_featnames(model)
    inputs = load(input_path)['x']
    contains = None
    for feat_set_name in list(inputs.keys()):
        cols = featnames['offer'] if 'offer' in feat_set_name else featnames[feat_set_name]
        inputs_df = pd.DataFrame(data=inputs[feat_set_name],
                                 index=index,
                                 columns=cols)
        if contains is None:
            full_lstgs = inputs_df.index.get_level_values('lstg')
            contains = full_lstgs.isin(lstgs)
        inputs_df = inputs_df.loc[contains, :]
        inputs[feat_set_name] = inputs_df
    return inputs


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


def load_reindex(part=None, name=None, lstgs=None):
    df = load_file(part, name, agent=False)
    df = df.reindex(index=lstgs, level='lstg')
    if name == 'x_offer':
        df[CENSORED] = df[EXP] & (df[DELAY] < 1)
    return df




