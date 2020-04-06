"""
Generate listing chunks used to periodically evaluate the RL
during training and update learning rate
"""
import argparse
from constants import VALIDATION, RL_EVAL_DIR, SIM_CHUNKS
from agent.agent_utils import remove_unlikely_arrival_lstgs
from rlenv.env_utils import get_env_sim_dir, load_chunk, dump_chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', required=True, type=int, help='chunk number')
    num = parser.parse_args().num
    num = ((num-1) % SIM_CHUNKS) + 1

    x_lstg, lookup = load_chunk(base_dir=get_env_sim_dir(VALIDATION), num=num)
    x_lstg, lookup = remove_unlikely_arrival_lstgs(x_lstg=x_lstg, lookup=lookup)
    path = '{}{}.gz'.format(RL_EVAL_DIR, num)
    dump_chunk(x_lstg=x_lstg, lookup=lookup, path=path)


if __name__ == '__main__':
    main()
