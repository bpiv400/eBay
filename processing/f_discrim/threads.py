import pandas as pd
from processing.processing_utils import input_partition, load_file, \
	get_days_delay, get_norm, get_x_thread, init_x, collect_date_clock_feats
from processing.f_discrim.discrim_utils import concat_sim_chunks, save_discrim_files
from utils import is_split
from constants import MONTH, IDX, SLR_PREFIX
from featnames import CON, NORM, SPLIT, DAYS, DELAY, EXP, AUTO, REJECT, CENSORED, \
	MONTHS_SINCE_LSTG, TIME_FEATS, MSG


def get_x_offer(offers, idx):
	# initialize dictionary of offer features
	x_offer = {}
	# dataframe of offer features for relevant threads
	offers = pd.DataFrame(index=idx).join(offers)
	# turn features
	for i in range(1, 8):
		# offer features at turn i
		offer = offers.xs(i, level='index').reindex(
			index=idx, fill_value=0).astype('float32')
		# set censored time feats to zero
		if i > 1:
			censored = (offer[EXP] == 1) & (offer[DELAY] < 1)
			offer.loc[censored, TIME_FEATS] = 0.0
		# drop time feats that are zero
		if i == 1:
			for feat in [DAYS, DELAY, REJECT]:
				assert (offer[feat].min() == 0) and (offer[feat].max() == 0)
				offer.drop(feat, axis=1, inplace=True)
		if i % 2 == 1:
			for feat in [AUTO]:
				assert (offer[feat].min() == 0) and (offer[feat].max() == 0)
				offer.drop(feat, axis=1, inplace=True)
		if i == 7:
			for feat in [MSG, SPLIT]:
				assert (offer[feat].min() == 0) and (offer[feat].max() == 0)
				offer.drop(feat, axis=1, inplace=True)
		# put in dictionary
		x_offer['offer%d' % i] = offer.astype('float32')
	return x_offer


def process_offers_sim(df, offer_cols):
	# clock features
	df = df.join(collect_date_clock_feats(df.clock))
	# days and delay
	df[DAYS], df[DELAY] = get_days_delay(df.clock.unstack())
	# concession as a decimal
	df.loc[:, CON] /= 100
	# indicator for split
	df[SPLIT] = df[CON].apply(lambda x: is_split(x))
	# total concession
	df[NORM] = get_norm(df[CON])
	# reject auto and exp are last
	df[REJECT] = df[CON] == 0
	df[AUTO] = (df[DELAY] == 0) & df.index.isin(IDX[SLR_PREFIX], level='index')
	df[EXP] = (df[DELAY] == 1) | df[CENSORED]
	# reorder columns to match observed
	df = df[offer_cols]
	return df


def process_threads_sim(part, df, thread_cols):
	# convert clock to months_since_lstg
	df = df.join(load_file(part, 'lookup').start_time)
	df[MONTHS_SINCE_LSTG] = (df.clock - df.start_time) / MONTH
	df = df.drop(['clock', 'start_time'], axis=1)
	# reorder columns to match observed
	df = df[thread_cols]
	return df


def process_sim(part, thread_cols, offer_cols):
	# construct inputs from simulations
	threads_sim, offers_sim = concat_sim_chunks(part)
	# conform to observed inputs
	threads_sim = process_threads_sim(part, threads_sim, thread_cols)
	offers_sim = process_offers_sim(offers_sim, offer_cols)
	# input features
	x = construct_x(part, threads_sim, offers_sim)
	return x


def construct_x(part, threads, offers):
	# master index
	idx = threads.index
	# initialize input dictionary with lstg features
	x = init_x(part, idx, drop_slr=False)
	# add thread features to x['lstg']
	x['lstg'] = pd.concat([x['lstg'], get_x_thread(threads, idx)], axis=1)
	# offer features
	x.update(get_x_offer(offers, idx))
	return x


def process_obs(part):
	# load inputs from data
	threads_obs = load_file(part, 'x_thread')
	offers_obs = load_file(part, 'x_offer')
	# dictionary of input features
	x_obs = construct_x(part, threads_obs, offers_obs)
	# return input features and dataframe columns
	return x_obs, threads_obs.columns, offers_obs.columns


def main():
	# extract partition from command line
	part = input_partition()
	print('%s/threads' % part)

	# observed data
	x_obs, thread_cols, offer_cols = process_obs(part)

	# simulated data
	x_sim = process_sim(part, thread_cols, offer_cols)

	# save data
	save_discrim_files(part, 'threads', x_obs, x_sim)


if __name__ == '__main__':
	main()