import pandas as pd
import torch
from processing.a_feats.tf import get_lstg_time_feats

# consts
COLS = ['accept', 'clock', 'reject', 'byr',
        'start_price', 'norm', 'price', 'censored', 'message']
ACCEPTANCE = 'acceptance' # time trigger for df not needed for environment


def compare_all(events, exp, time_checks, lstg=0):
    events = get_lstg_time_feats(events, full=False)
    act = [events.loc[(lstg, idx[0], idx[1])] for idx in time_checks]
    for curr_act, curr_exp, idx in zip(act, exp, time_checks):
        print('thread : {}, time: {}'.format(idx[0], idx[1]))
        print('curr processing feats')
        print(curr_act)
        print('curr time feats')
        print(curr_exp)
        compare(curr_act.values, curr_exp)


def compare(actual, exp):
    """
    Approximate equality between expected tensor and actual tensor

    :param actual: 1 dimensional torch.tensor
    :param exp: 1 dimensional torch.tensor
    :return: NA
    """
    if not isinstance(actual, torch.Tensor):
        actual = torch.from_numpy(actual).float()
    if not isinstance(exp, torch.Tensor):
        exp = torch.from_numpy(exp).float()
    assert torch.all(torch.lt(torch.abs(torch.add(actual, -exp)), 1e-6))


def get_exp_feats(idx, timefeats, exp, time_checks):
    print(idx)
    new_feats = timefeats.get_feats(thread_id=idx[0], time=idx[1])
    exp.append(new_feats)
    print(new_feats)
    time_checks.append(idx)


def update(events=None, timefeats=None, offer=None):
    """
    Updates events DataFrame and TimeFeatures object with the same event
    :param events: pd.DataFrame containing events
    :param timefeats: instance of rlenv.TimeFeatures.TimeFeatures
    :param offer: dictionary containing parameters of offer
    :return: updated events df
    """
    events = add_event(events, offer=offer)
    if timefeats is not None:
        timefeats.update_features(offer=offer)
    return events


def add_event(df, offer=None, lstg=1):
    """
    Adds an event to an events dataframe using the same dictionary of offer features
    as that TimeFeatures.update_feat
    :param df: dataframe containing events up to this point
    :param Offer offer: object giving information about the recent event
    :param int lstg: lstg id
    :return: updated dataframe
    """
    event = {}
    if df is None:
        df = pd.DataFrame(columns=COLS)
        df.index = pd.MultiIndex(levels=[[], [], []],
                                 codes=[[], [], []],
                                 names=['lstg', 'thread', 'index'])
    if lstg in df.index.levels[0]:
        last_index = df.xs(lstg, level='lstg', drop_level=True)
        if (lstg, offer.thread_id, 1) in df.index:
            last_index = last_index.xs(offer.thread_id, level='thread',
                                       drop_level=True).reset_index()['index'].max() + 1
        else:
            last_index = 1
    else:
        last_index = 1
    offer_index = pd.MultiIndex.from_tuples([(lstg, offer.thread_id, last_index)],
                                            names=['lstg', 'thread', 'index'])

    # repurpose offer dictionary
    event['start_price'] = 0
    event['clock'] = offer.time
    event['byr'] = offer.player == 'byr'
    event['accept'] = offer.accept
    event['reject'] = offer.reject
    event['norm'] = offer.price
    event['censored'] = offer.censored
    for col in ['price', 'message']:
        event[col] = 0
    keys = list(event.keys())
    keys.sort()
    cols = COLS.copy()
    cols.sort()
    assert len(keys) == len(cols)
    assert all([key == col for key, col in zip(keys, cols)])

    offer_df = pd.DataFrame(data=event, index=offer_index)
    df = df.append(offer_df, verify_integrity=True, sort=True)
    return df
