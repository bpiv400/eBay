import argparse
from agent.util import get_run_dir, only_byr_agent, load_valid_data
from agent.const import DELTA_CHOICES
from constants import IDX
from featnames import CON, X_OFFER, INDEX, TEST, BYR, AUTO, SLR


def main():
    # agent params from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=float, choices=DELTA_CHOICES)
    delta = parser.parse_args().delta
    byr = delta is None

    run_dir = get_run_dir(delta=delta)
    data = load_valid_data(part=TEST, run_dir=run_dir)
    if byr:
        data = only_byr_agent(data)

    turns = IDX[BYR] if byr else IDX[SLR]
    for turn in turns:
        print('Turn {}'.format(turn))
        con = data[X_OFFER][CON]
        if not byr:
            con = con.loc[~data[X_OFFER][AUTO]]
        con = con.xs(turn, level=INDEX)
        con_rate = con.groupby(con).count() / len(con)
        con_rate = con_rate[con_rate > .001]
        print(con_rate)


if __name__ == '__main__':
    main()
