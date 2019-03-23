"""
To create a time-valued feature:
1. Write a function to compute the feature for some time step below using
the following format
    Inputs:
        offer: series giving features for the offer associated with this timestep (list below).
        If the current time step isn't associated with an offer (i.e. the start and end of a lstg),
        offer is None.
        
        prev_feats: Series giving the time features at the previous step. For any given isolation level,
        the previous timestep will have all more granular features calculated
        
        edges: dictionary containing booleans for determining whether the current time step
        signals the start or end of a lstg or thread.
        The keys are (start_lstg, end_lstg, start_thread, end_thread). All mean exactly what you think

    Features in offer:
        'thread', 'index', 'clock', 'price', 'message', 'lstg', 'meta', 'leaf',
       'product', 'title', 'cndtn', 'slr', 'start_date', 'end_date',
       'relisted', 'fdbk_score', 'fdbk_pstv', 'start_price', 'photos',
       'slr_lstgs', 'slr_bos', 'views', 'wtchrs', 'sale_price', 'ship_slow',
       'ship_fast', 'ship_chosen', 'decline_price', 'accept_price', 'bin_rev',
       'store', 'slr_us', 'byr', 'byr_us', 'byr_hist', 'slr_hist',
       'start_time', 'flag'] 
       #NOTE: Let me know which of these we absolutely won't need. I suspect
       you won't need any of the constant value features for computing time-vals,
       but I'm not 100% sure. E.g. if we want a feature for lowest seller offer
       thus far, we might want to initialize it to the start_price

2. Once you have written the function in this script, go to time_feats and
add the function to the dictionary corresponding to the granualarity you would
like the feature calculated for. The format of each dictionary is:
    feature_name: (function_name, default value)
"""

import pandas as pd
import numpy as np


def lstg_max(offer=None, prev_feats=None, edges=None):
    """
    Names feature lstg_max
    """
    # if its thfe first offer return 0
    if offer is None and prev_feats is None:
        return 0
    else:
        if offer is None or offer['index'] % 2 == 0:
            return prev_feats['lstg_max']
        elif prev_feats is None:
            return offer['price']
        else:
            print(prev_feats.index)
            return max(prev_feats['lstg_max'], offer['price'])


def slr_open(offer=None, prev_feats=None, edges=None):
    """
    Computes a time-valued feature at the seller level to compute
    the total number of open listings for the slr

    Names feature slr_open
    """
    prev = 0 if prev_feats is None else prev_feats['slr_open']
    if edges['start_lstg']:
        return prev + 1
    elif edges['end_lstg']:
        return prev - 1
    else:
        return prev


def item_open(offer=None, prev_feats=None, edges=None):
    """
    Computes a time-valued feature at the item level to compute
    the total number of open listings corresponding to the item
    
    Names feature slr_open
    """
    prev = 0 if prev_feats is None else prev_feats['item_open']
    if edges['start_lstg']:
        return prev + 1
    elif edges['end_lstg']:
        return prev - 1
    else:
        return prev
