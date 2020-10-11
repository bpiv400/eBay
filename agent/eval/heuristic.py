import argparse
import torch
from agent.eval.AgentGenerator import AgentGenerator
from agent.models.HeuristicByr import HeuristicByr
from agent.models.HeuristicSlr import HeuristicSlr
from rlenv.generate.util import process_sims
from utils import run_func_on_chunks, process_chunk_worker, get_role
from constants import AGENT_PARTITIONS, HEURISTIC_DIR


def main():
    # parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=AGENT_PARTITIONS)
    parser.add_argument('--byr', action='store_true')
    args = parser.parse_args()

    # recreate model
    model = HeuristicByr() if args.byr else HeuristicSlr()

    # run in parallel on chunks
    sims = run_func_on_chunks(
        f=process_chunk_worker,
        func_kwargs=dict(
            part=args.part,
            gen_class=AgentGenerator,
            gen_kwargs=dict(
                model=model,
                byr=args.byr,
                slr=not args.byr,
            )
        )
    )

    # combine and process output
    output_dir = HEURISTIC_DIR + '{}/{}/'.format(get_role(args.byr), args.part)
    process_sims(part=args.part,
                 sims=sims,
                 output_dir=output_dir,
                 byr=args.byr,
                 save_inputs=False)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('forkserver')
    main()
