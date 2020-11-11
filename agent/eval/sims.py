import argparse
import os
from agent.eval.AgentGenerator import AgentGenerator
from agent.models.AgentModel import load_agent_model
from agent.models.HeuristicByr import HeuristicByr
from agent.models.HeuristicSlr import HeuristicSlr
from agent.util import get_paths, get_log_dir
from rlenv.generate.util import process_sims
from utils import compose_args, topickle, run_func_on_chunks, process_chunk_worker
from agent.const import AGENT_PARAMS, HYPER_PARAMS
from constants import DROPOUT_GRID, NUM_CHUNKS
from featnames import BYR, DROPOUT, TEST, AGENT_PARTITIONS


def main():
    # parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str,
                        choices=AGENT_PARTITIONS, default=TEST)
    parser.add_argument('--num', type=int, choices=range(1, NUM_CHUNKS + 1))
    parser.add_argument('--heuristic', action='store_true')
    parser.add_argument('--suffix', type=str)
    compose_args(arg_dict=AGENT_PARAMS, parser=parser)
    compose_args(arg_dict=HYPER_PARAMS, parser=parser)
    args = vars(parser.parse_args())

    # run directory
    if args['heuristic']:
        run_dir = get_log_dir(**args) + 'heuristic/'
    else:
        args[DROPOUT] = DROPOUT_GRID[args[DROPOUT]]
        _, _, run_dir = get_paths(**args)

    # (re)create model
    if args['heuristic']:
        model = HeuristicByr() if args[BYR] else HeuristicSlr()
    else:
        model_args = {BYR: args[BYR], 'value': False}
        model = load_agent_model(model_args=model_args, run_dir=run_dir)

    # generator
    if args['num'] is not None:
        # create output folder
        output_dir = run_dir + '{}/outcomes/'.format(args['part'])
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        # check if chunk has already been processed
        chunk = args['num'] - 1
        path = output_dir + '{}.pkl'.format(chunk)
        if os.path.isfile(path):
            print('Chunk {} already exists.'.format(chunk))
            exit(0)

        # process one chunk
        gen = AgentGenerator(model=model, byr=args[BYR], slr=not args[BYR])
        df = gen.process_chunk(part=args['part'], chunk=chunk)

        # save
        topickle(df, path)

    else:
        # run in parallel on chunks
        sims = run_func_on_chunks(
            f=process_chunk_worker,
            func_kwargs=dict(
                part=args['part'],
                gen_class=AgentGenerator,
                gen_kwargs=dict(
                    model=model,
                    byr=args[BYR],
                    slr=not args[BYR],
                )
            )
        )

        # combine and process output
        output_dir = run_dir + '{}/'.format(args['part'])
        process_sims(part=args['part'],
                     sims=sims,
                     output_dir=output_dir,
                     byr=args[BYR])


if __name__ == '__main__':
    main()
