from assess.util import load_data, get_valid_slr, find_best_run, get_lookup
from constants import TEST
from featnames import SLR, OBS, SIM, RL, NORM, MSG, DELAY, MONTHS_SINCE_LSTG, \
    BYR_HIST, TIME_FEATS, START_PRICE


def run_test(data=None, lookup=None):
    data, lookup = get_valid_slr(data=data, lookup=lookup)  # restrict to valid listings
    threads, offers = [data[k] for k in ['threads', 'offers']]

    msg2 = offers[MSG].xs(2, level='index').astype(bool)
    delay2 = offers[DELAY].xs(2, level='index')
    norm = offers[NORM].unstack()
    norm3 = norm.loc[norm[2] == 0., 3]
    norm3 = norm3[~norm3.isna()]
    norm3 = norm3[(delay2 < 1) & ~msg2]

    idx = norm3.index
    delay2 = delay2.reindex(index=idx)
    delay3 = offers[DELAY].xs(3, level='index').reindex(index=idx)
    threads_idx = threads.loc[idx]

    print('\tOffer count: {}'.format((norm3 == 1).sum()))
    print('\tBIN rate: {}'.format((norm3 == 1).mean()))
    print('\tAvg months (BIN): {}'.format(
        threads_idx.loc[norm3 == 1, MONTHS_SINCE_LSTG].mean()))
    print('\tAvg months (~BIN): {}'.format(
        threads_idx.loc[norm3 < 1, MONTHS_SINCE_LSTG].mean()))
    print('\tAvg hist (BIN): {}'.format(
        threads_idx.loc[norm3 == 1, BYR_HIST].mean()))
    print('\tAvg hist (~BIN): {}'.format(
        threads_idx.loc[norm3 < 1, BYR_HIST].mean()))
    print('\tAvg slr delay: {}'.format(delay2.mean()))
    print('\tAvg byr delay: {}'.format(delay3.mean()))
    print('\tAvg list price: {}'.format(
        lookup[START_PRICE].reindex(index=idx, level='lstg').mean()))

    print(offers.xs(3, level='index').loc[idx, TIME_FEATS].mean())


def main():
    lookup, filename = get_lookup(prefix=SLR)
    run_dir = find_best_run()

    data = dict()

    # observed sellers
    data[OBS] = load_data(part=TEST, lstgs=lookup.index, obs=True)
    data[SIM] = load_data(part=TEST, lstgs=lookup.index, sim=True)
    data[RL] = load_data(part=TEST, lstgs=lookup.index, run_dir=run_dir)

    for k, v in data.items():
        print(k)
        run_test(data=v, lookup=lookup)


if __name__ == '__main__':
    main()
