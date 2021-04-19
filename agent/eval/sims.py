import os
from agent.eval.AgentGenerator import SellerGenerator, BuyerGenerator
from agent.models.AgentModel import load_agent_model
from agent.models.heuristics import HeuristicSlr, HeuristicByr
from agent.eval.util import sim_args
from agent.util import get_run_dir, get_output_dir
from utils import topickle


def main():
    args = sim_args(num=True)

    # output directory
    output_dir = get_output_dir(**vars(args))
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

    # model
    if args.heuristic:
        model_cls = HeuristicByr if args.byr else HeuristicSlr
        model = model_cls(delta=args.delta)
    else:
        run_dir = get_run_dir(byr=args.byr,
                              delta=args.delta,
                              turn_cost=args.turn_cost)
        model_args = dict(byr=args.byr, value=False)
        model = load_agent_model(model_args=model_args, run_dir=run_dir)

    # generator
    if args.byr:
        gen = BuyerGenerator(model=model, agent_thread=args.agent_thread)
    else:
        gen = SellerGenerator(model=model)

    # process one chunk
    df = gen.process_chunk(part=args.part, chunk=chunk)

    # save
    topickle(df, path)


if __name__ == '__main__':
    main()
