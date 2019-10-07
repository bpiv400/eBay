<<<<<<< HEAD
import argparse, random, sys
=======
>>>>>>> 4448e0b7a3b2f9cca2b4f5395d4fd9517d672c4e
from compress_pickle import load, dump
import pandas as pd
import argparse, sys

sys.path.append('repo/')
from constants import *


if __name__ == "__main__":
	# partition number from command line
	parser = argparse.ArgumentParser()
	parser.add_argument('--num', action='store', type=int, required=True)
	num = parser.parse_args().num

<<<<<<< HEAD
def multiply_indices(s):
    # initialize arrays
    k = len(s.index.names)
    arrays = np.zeros((s.sum(),k+1), dtype=np.int64)
    count = 0
    # outer loop: range length
    for i in range(1, max(s)+1):
        index = s.index[s == i].values
        if len(index) == 0:
            continue
        # cartesian product of existing level(s) and period
        if k == 1:
            f = lambda x: cartesian([[x], list(range(i))])
        else:
            f = lambda x: cartesian([[e] for e in x] + [list(range(i))])
        # inner loop: rows of period
        for j in range(len(index)):
            arrays[count:count+i] = f(index[j])
            count += i
    # convert to multi-index
    return pd.MultiIndex.from_arrays(np.transpose(arrays), 
        names=s.index.names + ['period'])


def parse_days(diff, t0, t1):
    # count of arrivals by day
    days = diff.dt.days.rename('period').to_frame().assign(count=1)
    days = days.groupby(['lstg', 'period']).sum().squeeze().astype(np.uint8)
    # end of listings
    end = (pd.to_timedelta(t1 - t0, unit='s').dt.days).rename('period')
    end.loc[end > MAX_DAYS] = MAX_DAYS
    # create multi-index from end stamps
    idx = multiply_indices(end+1)
    # expand to new index and return
    return days.reindex(index=idx, fill_value=0).sort_index()


def get_y_arrival(lstgs, threads):
    d = {}
    # time_stamps
    t0 = lstgs.start_date * 24 * 3600
    t1 = lstgs.end_time
    diff = pd.to_timedelta(threads.clock - t0, unit='s')
    # append arrivals to end stamps
    d['days'] = parse_days(diff, t0, t1)
    return d

=======
	part = PARTITIONS[num-1]
>>>>>>> 4448e0b7a3b2f9cca2b4f5395d4fd9517d672c4e

	path = 'data/partitions/%s/x_offer.gz' % part

<<<<<<< HEAD
    # partition
	partitions = load(PARTS_DIR + 'partitions.gz')
	part = list(partitions.keys())[num-1]
	idx = partitions[part]
	path = lambda name: PARTS_DIR + part + '/' + name + '.gz'

    # load data and 
    lstgs = pd.read_csv(CLEAN_DIR + 'listings.csv', 
    	usecols=['lstg', 'start_date', 'end_time']).set_index(
    	'lstg').reindex(index=idx)
    threads = load_frames('threads').reindex(index=idx, level='lstg')
=======
	x_offer = load(path)

	x_offer.loc[x_offer.norm.isna(), 'norm'] = 0
>>>>>>> 4448e0b7a3b2f9cca2b4f5395d4fd9517d672c4e

	dump(x_offer, path)