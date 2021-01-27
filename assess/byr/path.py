from agent.util import only_byr_agent, load_valid_data, get_output_dir
from featnames import X_OFFER, CON, TEST


def feats_from_con(con=None):
    d = {'ends': con == -1, 'acc': con == 1, 'rej': con == 0}
    d['counter'] = ~d['acc'] & ~d['rej'] & ~d['ends']
    print('\tListing ends: {}'.format(d['ends'].mean()))
    print('\tSeller accepts: {}'.format(d['acc'].mean()))
    print('\tSeller counters: {}'.format(d['counter'].mean()))
    print('\tSeller rejects: {}'.format(d['rej'].mean()))
    return d


def main():
    run_dir = get_output_dir(part=TEST, heuristic=True)
    data = only_byr_agent(load_valid_data(part=TEST, run_dir=run_dir))
    con = data[X_OFFER][CON].unstack()
    con[con.isna()] = -1

    # first offer is half of list price
    assert all(con[1] == .5)

    # turn 2
    print('Turn 2')
    feats2 = feats_from_con(con=con[2])

    # turn 4
    idx4 = con[con[3] > -1].index
    feats4 = dict()
    for k in ['counter', 'rej']:
        print('Turn 4 after turn 2 {}'.format(k))
        mask = feats2[k].loc[idx4]
        idx = mask[mask].index
        feats4[k] = feats_from_con(con=con[4].loc[idx])

    # turn 6
    for k2 in ['counter', 'rej']:
        s = con[5].loc[feats2[k2][feats2[k2]].index]
        idx6 = s[s > -1].index
        for k4 in ['counter', 'rej']:
            print('Turn 6 after turn 2 {} & turn 4 {}'.format(k2, k4))
            mask = feats4[k2][k4].loc[idx6]
            idx = mask[mask].index
            _ = feats_from_con(con=con[6].loc[idx])


if __name__ == '__main__':
    main()
