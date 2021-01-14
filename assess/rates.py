import argparse
from agent.util import get_run_dir, only_byr_agent, load_valid_data
from utils import compose_args
from agent.const import AGENT_PARAMS
from constants import IDX
from featnames import CON, X_OFFER, INDEX, TEST, BYR, AUTO, SLR


def main():
    # agent params from command line
    parser = argparse.ArgumentParser()
    compose_args(arg_dict=AGENT_PARAMS, parser=parser)
    params = vars(parser.parse_args())

    run_dir = get_run_dir(**params)
    data = load_valid_data(part=TEST, run_dir=run_dir, byr=params[BYR])
    if params[BYR]:
        data = only_byr_agent(data)

    turns = IDX[BYR] if params[BYR] else IDX[SLR]
    for turn in turns:
        print('Turn {}'.format(turn))
        con = data[X_OFFER][CON]
        if not params[BYR]:
            con = con.loc[~data[X_OFFER][AUTO]]
        con = con.xs(turn, level=INDEX)
        con_rate = con.groupby(con).count() / len(con)
        con_rate = con_rate[con_rate > .001]
        print(con_rate)


if __name__ == '__main__':
    main()
