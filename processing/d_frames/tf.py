from compress_pickle import load, dump
import numpy as np, pandas as pd
from processing.processing_utils import input_partition, get_partition, load_frames
from processing.processing_consts import CLEAN_DIR


def remove_zeros(tf):
	toDrop = (tf.min(axis=1) == 0) & (tf.max(axis=1) == 0) 
	return tf[~toDrop]


def remove_lstg_end(tf, lstg_end):
	tf.reset_index('clock', inplace=True)
	if len(tf.index.names) == 1:
		end = lstg_end.reindex(index=tf.index)
	else:
		end = lstg_end.reindex(index=tf.index, level='lstg')
	return tf[tf.clock < end]


if __name__ == "__main__":
	# data partition
	part = input_partition()
	idx, path = get_partition(part)

	# end of listing
	lstg_end = load(CLEAN_DIR + 'listings.pkl').end_time.reindex(index=idx)

	# arrival time feats
	tf_arrival = load_frames('tf_arrival').reindex(
		index=idx, level='lstg')

	tf_arrival = remove_zeros(tf_arrival)
	tf_arrival = remove_lstg_end(tf_arrival, lstg_end)

	dump(tf_arrival, path('tf_arrival'))

	# delay time feats
	tf_delay = load_frames('tf_delay').reindex(
		index=idx, level='lstg').drop('index', axis=1)

	tf_delay.index.set_levels(tf_delay.index.levels[2].astype('int8'), 
		level=2, inplace=True)

	tf_delay = remove_zeros(tf_delay)
	tf_delay = remove_lstg_end(tf_delay, lstg_end)

	dump(tf_delay, path('tf_delay'))

	# offer time feats
	tf_offer = load_frames('tf_offer').reindex(
		index=idx, level='lstg')
	dump(tf_offer, path('tf_offer'))