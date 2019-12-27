from compress_pickle import dump
import numpy as np, pandas as pd
from processing.processing_utils import input_partition, get_partition, load_frames


if __name__ == "__main__":
	# data partition
	part = input_partition()
	idx, path = get_partition(part)

	# arrival time feats
	tf_arrival = load_frames('tf_arrival').reindex(
		index=idx, level='lstg')
	dump(tf_arrival, path('tf_arrival'))

	# delay time feats
	tf_delay = load_frames('tf_delay').reindex(
        index=idx, level='lstg').drop('index', axis=1)
	dump(tf_delay, path('tf_delay'))

	# offer time feats
	tf_offer = load_frames('tf_offer').reindex(
		index=idx, level='lstg')
	dump(tf_offer, path('tf_offer'))