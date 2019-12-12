"""
Assumptions:
    1. All byr w2v features and no others appear in 'w2v_byr'
    2. All slr w2v features and no others appear in 'w2v_slr'
    3. All condition-level features and no others appear in 'cndtn'
    4. ALl slr-level features and no others appear in 'slr'
    5. All cat-level features and no others appear in 'cat'
"""

# imports
from copy import deepcopy
from pickle import dump
import pandas as pd
from rlenv.env_consts import AGENT_FEATS_FILENAME
from constants import FEATNAMES_DIR
from rlenv.env_utils import load_featnames

# constants
COVER_PAGE = 'feats'
# columns in excel
LSTG_COL = 'LSTG'
THREAD_COL = 'THREAD'
BYR1_COL = 'BYR_1'
SLR6_COL = 'SLR_6'
BYR7_COL = 'BYR_7'
CURR_COL = 'CURR'
SLR_COL = 'SLR_TURN'
BYR_COL = 'BYR_TURN'
PREV_COL = 'PREV'
# names in featnames
W2V_BYR = 'byr#'
W2V_SLR = 'slr#'
CAT = 'cat_[featname]'
CNDTN = 'cndtn_[featname]'
SLR = 'slr_[featname]'
# embeddings grouping names
GROUPS = {
    W2V_SLR: 'w2v_slr',
    W2V_BYR: 'w2v_byr',
    CAT: 'cat',
    CNDTN: 'cndtn',
}


def parse_lstg_feats(lstg_ser, feats):
    tf = [feat[4:] for feat in feats['x']['cat']]
    lstg_ser = ser2list(lstg_ser)
    for shorthand, group_name in GROUPS.items():
        if shorthand not in lstg_ser:
            del feats['x'][group_name]
        else:
            lstg_ser.remove(shorthand)
            for feat in feats['x'][group_name]:
                lstg_ser.append(feat)
    if SLR in lstg_ser:
        for feat in tf:
            lstg_ser.append('slr_{}'.format(feat))
    return lstg_ser


def ser2list(candidate):
    if isinstance(candidate, pd.Series):
        candidate = candidate.loc[~candidate.isna()]
        candidate = list(candidate.values)
    if not isinstance(candidate, list):
        raise RuntimeError('candidate must be list or series')
    return candidate


def add_feats(base, candidate):
    candidate = ser2list(candidate)
    for feat in candidate:
        base.append(feat)


def parse_turn(base, candidate, turn):
    candidate = ser2list(candidate)
    candidate = ['{}_{}'.format(feat, turn) for feat in candidate]
    add_feats(base, candidate)


def parse_feats(df, agent, feats):
    byr = 'byr' in agent
    all_feats = parse_lstg_feats(df[LSTG_COL], feats)
    add_feats(all_feats, df[THREAD_COL])
    # remove offer7 from featnames and parse special turn 6
    # column if slr model
    if not byr:
        del feats['x']['offer7']
        parse_turn(all_feats, df[SLR6_COL], 6)
    # otherwise parse buyer turn 7 column and normal seller turn 6
    else:
        parse_turn(all_feats, df[BYR7_COL], 7)
        parse_turn(all_feats, df[SLR_COL], 6)
    # add first buyer turn
    parse_turn(all_feats, df[BYR1_COL], 1)
    # add remaining turns
    for byr_turn in [3, 5]:
        parse_turn(all_feats, deepcopy(df[BYR_COL]), byr_turn)
        parse_turn(all_feats, deepcopy(df[SLR_COL]), byr_turn - 1)
    # add turn indicators
    if not byr:
        add_feats(all_feats, ['t1', 't2'])
    else:
        add_feats(all_feats, ['t1', 't2', 't3'])
    # error checking
    check_foreign(all_feats, feats['x'], byr=byr)
    # prune dictionary of featnames based on list of all features
    all_feats = set(all_feats)
    featnames = prune_feats(all_feats, feats['x'])
    # add current time/clock features
    curr = ser2list(df[CURR_COL])
    featnames['x']['curr'] = curr
    # add previous time/clock features
    prev = ser2list(df[PREV_COL])
    prev = ['{}_prev'.format(feat) for feat in prev]
    featnames['x']['prev'] = prev
    delay_featnames = prune_for_delay(deepcopy(featnames['x']), byr=byr)
    return featnames, delay_featnames


def prune_for_delay(agent_feats, byr=False):
    if byr:
        targ_turn = 7
    else:
        targ_turn = 6
    targ_feats = ['{}_{}'.format(feat, targ_turn) for feat in ['days', 'delay']]
    print(targ_feats)
    for set_name in agent_feats.keys():
        for feat in targ_feats:
            if feat in agent_feats[set_name]:
                agent_feats[set_name].remove(feat)
    return {'x': agent_feats}


def prune_feats(agent_feats, model_feats):
    pruned = dict()
    for set_name, feats in model_feats.items():
        pruned[set_name] = [feat for feat in feats if feat in agent_feats]
        if len(pruned[set_name]) == 0:
            raise RuntimeError('No features given to RL for {} embedding'.format(set_name))
    return {'x': pruned}


# checks whe    ther
def check_foreign(agent_feats, model_feats, byr=False):
    agent_feat_set = set(agent_feats)
    model_feat_set = set()
    for _, feats in model_feats.items():
        for feat in feats:
            model_feat_set.add(feat)
    foreign = agent_feat_set.difference(model_feat_set)
    if len(foreign) > 0:
        print('invaders: {}'.format(foreign))
        raise RuntimeError()
    if byr:
        turns = [1, 3, 5, 7]
    else:
        turns = [2, 4, 6]
    for turn in turns:
        if 'msg_{}'.format(turn) in agent_feat_set:
            raise RuntimeError('deterministic msg included')


def store_featnames(feat_sets):
    for agent, (featnames, delay_featnames) in feat_sets.items():
        no_delay_path = '{}{}.pkl'.format(FEATNAMES_DIR, agent)
        delay_path = '{}{}_d.pkl'.format(FEATNAMES_DIR, agent)
        dump(featnames, open(no_delay_path, 'wb'))
        dump(delay_featnames, open(delay_path, 'wb'))


def main():
    path = '{}{}'.format(FEATNAMES_DIR, AGENT_FEATS_FILENAME)
    f = pd.ExcelFile(path, None)
    content = pd.read_excel(f, None)
    con_byr = load_featnames('con_byr')
    feat_sets = dict()
    f.close()
    for agent, page in content.items():
        if agent != COVER_PAGE:
            featnames = parse_feats(page, agent, deepcopy(con_byr))
            feat_sets[agent] = featnames
    store_featnames(feat_sets)


if __name__ == '__main__':
    main()
