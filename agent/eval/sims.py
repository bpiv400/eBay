import os
from agent.eval.AgentGenerator import AgentGenerator
from agent.models.AgentModel import load_agent_model
from agent.models.HeuristicByr import HeuristicByr
from agent.models.HeuristicSlr import HeuristicSlr
from agent.eval.util import sim_args
from agent.util import get_run_dir, get_output_dir
from utils import topickle


def main():
    args = sim_args(num=True)
    byr = args.delta is None

    # output directory
    output_dir = get_output_dir(part=args.part,
                                heuristic=args.heuristic,
                                delta=args.delta)

    # create output folder
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # check if chunk has already been processed
    chunk = args.num - 1
    path = output_dir + 'outcomes/{}.pkl'.format(chunk)
    if os.path.isfile(path):
        print('Chunk {} already exists.'.format(chunk))
        exit(0)

    # generator
    gen_cls = AgentGenerator
    if args.heuristic:
        model = HeuristicByr() if byr else HeuristicSlr(delta=args.delta)
    else:
        run_dir = get_run_dir(delta=args.delta)
        model_args = dict(byr=byr, value=False)
        model = load_agent_model(model_args=model_args, run_dir=run_dir)
    gen = gen_cls(model=model, byr=byr)

    # process one chunk
    df = gen.process_chunk(part=args.part, chunk=chunk)

    # save
    topickle(df, path)


if __name__ == '__main__':
    main()
