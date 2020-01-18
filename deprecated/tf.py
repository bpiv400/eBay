from compress_pickle import load, dump
import numpy as np, pandas as pd
from processing.processing_utils import input_partition, get_partition, load_frames
from processing.processing_consts import CLEAN_DIR


if __name__ == "__main__":
	# data partition
	part = input_partition()
	idx, path = get_partition(part)

	# arrival time feats
	tf_arrival = load_frames('tf_arrival').reindex(
		index=idx, level='lstg')
	dump(tf_arrival, path('tf_arrival'))

	# offer time feats
	tf_offer = load_frames('tf_offer').reindex(
		index=idx, level='lstg')
	dump(tf_offer, path('tf_offer'))