import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import pickle
import math
from datetime import datetime as dt
import sys
import os

def get_offer_set(turn):
    '''
    Description: Generates a list of the offers that have been observed
    thusfar, including the current offer
    '''
    turn_num = int(turn[1])
    turn_type = turn[0]
    offer_set = []
    for in range(turn_num + 1):
        offer_set.append('offr_b' + str(i))
        if i < turn_num:
            offer_set.append('offr_s' + str(i))
        elif turn_type == 's':
            offer_set.append('offr_s' + str(i))
    offer_set.append('start_price_usd')
    return offer_set

def round_inds(df, round_vals, turn):
    '''
    Description: Create indicators for whether each offer in the data set 
    is exactly equal to some round value (create one such indicator for each 
    offer and round value pair). Additionally, for each offer in the data set, create
    an indicator for whether the offer is close (within 1%) but not equal to any of 
    the round values used to create indicators. Does create indicators for start_price_usd
    since this is grabbed by get_offer_set, BUT doesn't create indicators for the response
    offer
    Inputs:
        df: a pandas data frame containing offers
        round_vals: a list of values considered 'round', ie those that people may be 
        likely to converge to (eg [1, 5, 10, 25])
        turn: a string giving the name of the current turn
    Output: a pandas data frame containing indicators described above
    '''
    round_vals = [int(round_val) for round_val in round_vals]
    # get list of all offers in the data set except for the next
    # offer (ie response offer)
    offer_set = get_offer_set(turn)
    # iterate over round values
    for offr_name in offer_set:
        # iterate over all offers the data set except the next offr
        # create series to encode whether the offer in question
        # is near but not directly at a round value
        slack_ser = pd.Series(0, index=df.index)
        # name slack column for current offer
        slack_ser_name = 'slk_%s' % offr_name
        # create empty numpy arrays to track 
        # all of the round indices observed so far and all of the
        # slack indices because offers considered 'slack' must 
        # not be equal to any of the even values, not just the current one
        # so we track all slack inds and round inds over all iterations
        all_round_inds = np.array()
        all_slack_inds = np.array()
        for curr_val in round_vals:
            # create series for round indicator for current pair of 
            # offer and value
            curr_ser = pd.Series(0, index=df.index)
            # give a name to the current round series
            new_feat_name = 'rnd_%d_%s' % (curr_val, offr_name)
            # find all indices for rows where the current offer is a multiple of the current value
            offr_inds = df[df[offr_name] % curr_val == 0].index
            # activate the indicator for the se values
            curr_ser.loc[offr_inds] = 1
            # add these indices to the list of all indices found round so far for the
            # current offer
            all_round_inds = np.append(all_round_inds, offr_inds.values)
            # grab a series of the current offer consisting of rows where the 
            # offer in this iteration is nonzero because we must divide by this value
            # to determine slack indices -- which would create an NA headache that would
            # ruin my life for sure
            non_zero_offs = df.loc[df[df[offr_name] != 0].index, offr_name].copy()
            # find slack indices by finding the remainder of the current offer 
            # divided by the current value then dividing this value by the value of the
            # current offer and taking indices where this is below some threshold
            # arbitrary .01 for now
            slack_inds = non_zero_offs[((non_zero_offs % curr_val) / non_zero_offs) < .01].index
            # add these indices to the running list of slack indices for the current offer
            all_slack_inds = np.append(all_slack_inds, slack_inds.values)
            # finally, add the indicator for the current round-offer pair to the 
            # data frame under the name created for it previously
            df[new_feat_name] = curr_ser
        # find all of the slack indices that are not round indices
        slack_inds = np.setdiff1d(all_slack_inds, all_round_inds)
        # activate these indices in the slack series for the current offer
        slack_ser.loc[slack_inds] = 1
        # finally, add this series to the data frame with the name created for it 
        # above
        df[slack_ser_name] = slack_ser
    # return the data frame with the new features
    return df


def main():
    # parse parameters
    parser = argparse.ArgumentParser(
        description='associate threads with all relevant variables')
    # subdirectory name, corresponding to the type of file being pre-processed
    # (toy, train, test, etc.)
    # 'subdir-n.csv'
    parser.add_argument('--name', action='store', type=str)
    # turn name of the last offer made before the prediction variable
    # for this data set
    parser.add_argument('--turn', action='store', type=str)
    # name of the experiment
    parser.add_argument('--exp', action='store', type=str)
    # parse arguments
    args = parser.parse_args()
    filename = args.name
    subdir = args.dir
    turn = args.turn.strip()
    exp_name = args.exp

    read_loc = 'data/exps/' + exp_name '/' + 'turns/' + turn + '/' + filename
    df = pd.read_csv(read_loc, index_col=False)

    gen_offr_list 
if __name__ == '__main__':
    main()
    