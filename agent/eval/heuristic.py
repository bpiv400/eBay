import argparse
import torch
from agent.eval.AgentGenerator import AgentGenerator
from agent.models.HeuristicByr import HeuristicByr
from agent.models.HeuristicSlr import HeuristicSlr
from rlenv.generate.util import process_sims
from agent.util import get_log_dir
from utils import run_func_on_chunks, process_chunk_worker, compose_args
from agent.const import AGENT_PARAMS
from featnames import BYR, AGENT_PARTITIONS


def main():
    # parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=AGENT_PARTITIONS)
    compose_args(arg_dict=AGENT_PARAMS, parser=parser)
    args = vars(parser.parse_args())

    # recreate model
    model_cls = HeuristicByr if args[BYR] else HeuristicSlr()
    model = model_cls(**args)

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
    output_dir = get_log_dir(**args) + 'heuristic/{}/'.format(args['part'])
    process_sims(part=args['part'],
                 sims=sims,
                 output_dir=output_dir,
                 byr=args[BYR],
                 save_inputs=False)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('forkserver')
    main()
