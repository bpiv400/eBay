import pandas as pd
from constants import PARTS_DIR, DAY, WEEK, MAX_DAYS
from featnames import BYR_HIST, WEEKS_SINCE_LSTG, START_DATE
from processing.util import get_lstgs, load_feats, hist_to_pctile
from utils import topickle, get_weeks_since_lstg, input_partition


def create_x_thread(lstgs=None):
    # load data
    offers = load_feats('offers', lstgs=lstgs)
    thread_start = offers.clock.xs(1, level='index')
    start_date = load_feats('listings', lstgs=lstgs)[START_DATE]
    lstg_start = start_date.astype('int64') * 24 * 3600
    hist = load_feats('threads', lstgs=lstgs)[BYR_HIST]

    # months since lstg start
    weeks = get_weeks_since_lstg(lstg_start, thread_start)
    weeks = weeks.rename(WEEKS_SINCE_LSTG)
    assert weeks.max() < (MAX_DAYS * DAY) / WEEK

    # convert hist to percentile
    hist = hist_to_pctile(hist)

    # create dataframe
    x_thread = pd.concat([weeks, hist], axis=1)

    return x_thread


def main():
    part = input_partition()
    print('{}/x_thread'.format(part))

    x_thread = create_x_thread(lstgs=get_lstgs(part))
    topickle(x_thread, PARTS_DIR + '{}/x_thread.pkl'.format(part))


if __name__ == "__main__":
    main()
