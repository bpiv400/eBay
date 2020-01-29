import sys, os, argparse
import torch
import torch.nn.functional as F
import numpy as np, pandas as pd
from compress_pickle import load
from processing.f_discrim.discrim_utils import get_batches, PartialDataset, get_sim_times
from processing.processing_utils import load_file, get_arrival_times, get_interarrival_period, get_days_delay
from utils import load_model, load_featnames
from constants import VALIDATION, ENV_SIM_DIR, SIM_CHUNKS, INPUT_DIR, INDEX_DIR


def num_threads(df, lstgs):
	s = df.reset_index('thread')['thread'].groupby('lstg').count()
	s = s.reindex(index=lstgs, fill_value=0)
	s = s.groupby(s).count() / len(lstgs)
	return s


def num_offers(df):
	s = df.reset_index('index')['index'].groupby(['lstg', 'thread']).count()
	s = s.groupby(s).count() / len(s)
	return s


def avg_con(df):
	if df.con.max() == 100:
		df['con'] /= 100
	return df.con.groupby('index').mean()


def avg_msg(df):
	return df.msg.groupby('index').mean()


def avg_delay(df):
	if 'delay' not in df.columns:
		df['days'], df['delay'] = get_days_delay(df.clock.unstack())
	return df.delay.groupby('index').mean()


# simulation results
offers_sim, threads_sim = [], []
for i in range(1, SIM_CHUNKS+1):
	sim = load(ENV_SIM_DIR + '{}/discrim/{}.gz'.format(VALIDATION, i))
	offers_sim.append(sim['offers'])
	threads_sim.append(sim['threads'])

# concatenate and sort
offers_sim = pd.concat(offers_sim, axis=0).sort_index()
threads_sim = pd.concat(threads_sim, axis=0).sort_index()

# data
lstgs = load_file(VALIDATION, 'lookup').index
threads_obs = load_file(VALIDATION, 'x_thread')
offers_obs = load_file(VALIDATION, 'x_offer')

# drop censored observations
offers_sim = offers_sim[~offers_sim.censored].drop('censored', axis=1)
offers_obs = offers_obs[(offers_obs.delay == 1) | ~offers_obs.exp]

# number of threads per listing
threads_per_lstg = pd.concat([num_threads(threads_obs, lstgs).rename('obs'), 
							  num_threads(threads_sim, lstgs).rename('sim')], axis=1)
print(threads_per_lstg)

# number of offers per thread
offers_per_thread = pd.concat([num_offers(offers_obs).rename('obs'),
							   num_offers(offers_sim).rename('sim')], axis=1)
print(offers_per_thread)

# average delay by turn
delay_by_turn = pd.concat([avg_delay(offers_obs).rename('obs'),
						   avg_delay(offers_sim).rename('sim')], axis=1)
print(delay_by_turn)

# average concession by turn
con_by_turn = pd.concat([avg_con(offers_obs).rename('obs'),
						 avg_con(offers_sim).rename('sim')], axis=1)
print(con_by_turn)


# number of messages by turn
msg_by_turn = pd.concat([avg_msg(offers_obs).rename('obs'),
						 avg_msg(offers_sim).rename('sim')], axis=1)
print(msg_by_turn)