from inputs.util import save_files, get_x_thread, get_x_offer_init, get_ind_x
from utils import input_partition, load_file
from constants import IDX, SLR, CON_MULTIPLIER, AGENT_PARTITIONS, POLICY_SLR
from featnames import EXP, CON, DELAY, AUTO, LOOKUP


def construct_x(idx=None, threads=None, offers=None):
    # initialize dictionary with thread features
    x = {'thread': get_x_thread(threads, idx, turn_indicators=True)}

    # offer features
    x.update(get_x_offer_init(offers, idx, role=SLR))

    # no nans
    for v in x.values():
        assert v.isna().sum().sum() == 0

    return x


def get_y(df):
    # concession is an int from 0 to 100
    y = (df[CON] * CON_MULTIPLIER).astype('int8')
    # expired offer is last index
    y[df[EXP]] = CON_MULTIPLIER + 1
    return y


def create_index(offers):
    slr_turn = offers.index.isin(IDX[SLR], level='index')
    censored = offers[EXP] & (offers[DELAY] < 1)
    mask = slr_turn & ~offers[AUTO] & ~censored
    idx = offers[mask].index
    return idx


def process_inputs(part):
    # load dataframes
    lookup = load_file(part, LOOKUP, agent=True)
    threads = load_file(part, 'x_thread', agent=True)
    offers = load_file(part, 'x_offer', agent=True)

    # master index
    idx = create_index(offers)

    # outcome and master index
    y = get_y(offers.loc[idx, [CON, EXP]])

    # input features dictionary
    x = construct_x(idx=y.index,
                    threads=threads,
                    offers=offers)

    # indices for listing features
    idx_x = get_ind_x(lstgs=lookup.index, idx=idx)

    return {'y': y, 'x': x, 'idx_x': idx_x}


def main():
    # extract parameters from command line
    part = input_partition()
    print('{}/{}'.format(part, POLICY_SLR))

    # policy is trained on TRAIN_MODELS
    assert part in AGENT_PARTITIONS

    # input dataframes, output processed dataframes
    d = process_inputs(part)

    # save various output files
    save_files(d, part, POLICY_SLR)


if __name__ == '__main__':
    main()
