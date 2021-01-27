import os
from agent.eval.AgentGenerator import AgentGenerator
from agent.models.AgentModel import load_agent_model
from agent.models.heuristics import HeuristicSlr, HeuristicByr
from agent.eval.util import sim_args
from agent.util import get_run_dir, get_output_dir
from utils import topickle


def main():
    args = sim_args(num=True)

    # output directory
    output_dir = get_output_dir(part=args.part,
                                heuristic=args.heuristic,
                                byr=args.byr,
                                delta=args.delta)
    outcome_dir = output_dir + 'outcomes/'

    # create output folder
    if not os.path.isdir(outcome_dir):
        os.makedirs(outcome_dir)

    # check if chunk has already been processed
    chunk = args.num - 1
    path = outcome_dir + '{}.pkl'.format(chunk)
    if os.path.isfile(path):
        print('Chunk {} already exists.'.format(chunk))
        exit(0)

    # generator
    gen_cls = AgentGenerator
    if args.heuristic:
        model = HeuristicByr() if args.byr else HeuristicSlr(delta=args.delta)
    else:
        run_dir = get_run_dir(byr=args.byr, delta=args.delta)
        model_args = dict(byr=args.byr, value=False)
        model = load_agent_model(model_args=model_args, run_dir=run_dir)
    gen = gen_cls(model=model, byr=args.byr)

    # process one chunk
    df = gen.process_chunk(part=args.part, chunk=chunk)

    # save
    topickle(df, path)


if __name__ == '__main__':
    main()
