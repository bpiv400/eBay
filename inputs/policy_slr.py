from inputs.util import save_files, get_x_thread, get_x_offer_init, get_ind_x
from utils import input_partition, load_file
from constants import IDX, CON_MULTIPLIER, POLICY_SLR
from featnames import EXP, CON, DELAY, AUTO, LOOKUP, SLR


def construct_x(idx=None, threads=None, offers=None):
    # initialize dictionary with thread features
    x = {'thread': get_x_thread(threads, idx, turn_indicators=True)}
    # offer features
    x.update(get_x_offer_init(offers, idx, role=SLR))
    # no nans
    for v in x.values():
        assert v.isna().sum().sum() == 0
    return x


def create_index(offers):
    slr_turn = offers.index.isin(IDX[SLR], level='index')
    censored = offers[EXP] & (offers[DELAY] < 1)
    mask = slr_turn & ~offers[AUTO] & ~censored
    idx = mask[mask].index
    return idx


def process_inputs(part):
    # load dataframes
    lstgs = load_file(part, LOOKUP).index
    threads = load_file(part, 'x_thread')
    offers = load_file(part, 'x_offer')

    # master index
    idx = create_index(offers)

    # outcome and master index
    y = (offers.loc[idx, CON] * CON_MULTIPLIER).astype('int8')

    # input features dictionary
    x = construct_x(idx=idx, threads=threads, offers=offers)

    # indices for fixed features
    idx_x = get_ind_x(lstgs=lstgs, idx=idx)

    return {'y': y, 'x': x, 'idx_x': idx_x}


def main():
    # extract parameters from command line
    part = input_partition()
    print('{}/{}'.format(part, POLICY_SLR))

    # input dataframes, output processed dataframes
    d = process_inputs(part)

    # save various output files
    save_files(d, part, POLICY_SLR)


if __name__ == '__main__':
    main()
