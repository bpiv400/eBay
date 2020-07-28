from compress_pickle import dump
import pandas as pd
from constants import PARTS_DIR
from featnames import BYR_HIST, MONTHS_SINCE_LSTG, START_DATE
from processing.util import get_lstgs, load_feats, hist_to_pctile
from utils import get_months_since_lstg, input_partition


def create_x_thread(lstgs=None):
    # load data
    offers = load_feats('offers', lstgs=lstgs)
    thread_start = offers.clock.xs(1, level='index')
    start_date = load_feats('listings', lstgs=lstgs)[START_DATE]
    lstg_start = start_date.astype('int64') * 24 * 3600
    hist = load_feats('threads', lstgs=lstgs)[BYR_HIST]

    # months since lstg start
    months = get_months_since_lstg(lstg_start, thread_start)
    months = months.rename(MONTHS_SINCE_LSTG)
    assert months.max() < 1

    # convert hist to percentile
    hist = hist_to_pctile(hist)

    # create dataframe
    x_thread = pd.concat([months, hist], axis=1)

    return x_thread


def main():
    part = input_partition()
    print('{}/x_thread'.format(part))

    x_thread = create_x_thread(lstgs=get_lstgs(part))
    dump(x_thread, PARTS_DIR + '{}/x_thread.gz'.format(part))


if __name__ == "__main__":
    main()
