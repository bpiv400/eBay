import argparse
import torch
from agent.eval.AgentGenerator import AgentGenerator
from agent.models.AgentModel import load_agent_model
from agent.models.HeuristicByr import HeuristicByr
from agent.models.HeuristicSlr import HeuristicSlr
from agent.util import get_paths
from rlenv.generate.util import process_sims
from utils import run_func_on_chunks, process_chunk_worker, compose_args
from agent.const import AGENT_PARAMS, HYPER_PARAMS
from constants import AGENT_PARTITIONS, DROPOUT_GRID, TEST
from featnames import BYR, DROPOUT


def main():
    # parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=AGENT_PARTITIONS, default=TEST)
    parser.add_argument('--heuristic', action='store_true')
    parser.add_argument('--suffix', type=str)
    compose_args(arg_dict=AGENT_PARAMS, parser=parser)
    compose_args(arg_dict=HYPER_PARAMS, parser=parser)
    args = vars(parser.parse_args())

    # translate dropout index
    args[DROPOUT] = DROPOUT_GRID[args[DROPOUT]]

    # environment class and run directory
    _, _, run_dir = get_paths(**args)

    # recreate model
    if args['heuristic']:
        if args[BYR]:
            model = HeuristicByr()
        else:
            model = HeuristicSlr()
    else:
        model_args = {BYR: args[BYR], 'value': False}
        model = load_agent_model(model_args=model_args, run_dir=run_dir)

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
    if args['heuristic']:
        output_dir += 'heuristic/'
    process_sims(part=args['part'],
                 sims=sims,
                 output_dir=output_dir,
                 byr=args[BYR])


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('forkserver')
    main()
