import argparse
from agent.util import get_log_dir, get_paths
from rlenv.generate.util import process_sims
from utils import unpickle, compose_args
from agent.const import AGENT_PARAMS, HYPER_PARAMS
from constants import DROPOUT_GRID, NUM_CHUNKS
from featnames import AGENT_PARTITIONS, DROPOUT, TEST


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str,
                        choices=AGENT_PARTITIONS, default=TEST)
    parser.add_argument('--heuristic', action='store_true')
    parser.add_argument('--suffix', type=str)
    compose_args(arg_dict=AGENT_PARAMS, parser=parser)
    compose_args(arg_dict=HYPER_PARAMS, parser=parser)
    args = vars(parser.parse_args())

    # output directory
    if args['heuristic']:
        run_dir = get_log_dir(**args) + 'heuristic/'
    else:
        args[DROPOUT] = DROPOUT_GRID[args[DROPOUT]]
        _, _, run_dir = get_paths(**args)
    output_dir = run_dir + '{}/'.format(args['part'])

    # concatenate
    sims = []
    for i in range(NUM_CHUNKS):
        chunk_path = output_dir + 'outcomes/{}.pkl'.format(i)
        sims.append(unpickle(chunk_path))

    # clean and save
    process_sims(part=args['part'], sims=sims, output_dir=output_dir)


if __name__ == '__main__':
    main()
