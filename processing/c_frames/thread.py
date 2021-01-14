import pandas as pd
from processing.util import get_lstgs, feat_to_pctile
from utils import topickle, get_days_since_lstg, input_partition, load_feats
from constants import PARTS_DIR, MAX_DAYS, DAY
from featnames import BYR_HIST, DAYS_SINCE_LSTG, START_DATE


def create_x_thread(lstgs=None):
    # load data
    offers = load_feats('offers', lstgs=lstgs)
    thread_start = offers.clock.xs(1, level='index')
    start_date = load_feats('listings', lstgs=lstgs)[START_DATE]
    lstg_start = start_date.astype('int64') * DAY
    hist = load_feats('threads', lstgs=lstgs)[BYR_HIST]

    # days since lstg start
    days = get_days_since_lstg(lstg_start, thread_start)
    days = days.rename(DAYS_SINCE_LSTG)
    assert days.max() < MAX_DAYS

    # convert hist to percentile
    hist = feat_to_pctile(hist)

    # create dataframe
    x_thread = pd.concat([days, hist], axis=1)

    return x_thread


def main():
    part = input_partition()
    print('{}/x_thread'.format(part))

    x_thread = create_x_thread(lstgs=get_lstgs(part))
    topickle(x_thread, PARTS_DIR + '{}/x_thread.pkl'.format(part))


if __name__ == "__main__":
    main()
