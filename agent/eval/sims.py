import os
from agent.eval.AgentGenerator import AgentGenerator
from agent.models.AgentModel import load_agent_model
from agent.models.HeuristicByr import HeuristicByr
from agent.models.HeuristicSlr import HeuristicSlr
from agent.eval.util import sim_args
from agent.util import get_run_dir, get_output_dir
from rlenv.generate.util import process_sims
from utils import topickle, run_func_on_chunks, process_chunk_worker
from featnames import BYR


def main():
    args = sim_args(num=True)

    # output directory
    output_dir = get_output_dir(**args)

    # generator
    gen_cls = AgentGenerator
    if args['heuristic']:
        model_cls = HeuristicByr if args[BYR] else HeuristicSlr
        model = model_cls(**args)
    else:
        run_dir = get_run_dir(**args)
        model_args = {BYR: args[BYR], 'value': False}
        model = load_agent_model(model_args=model_args, run_dir=run_dir)
    gen_args = dict(model=model, byr=args[BYR])

    # generate
    if args['num'] is not None:
        # create output folder
        outcome_dir = output_dir + 'outcomes/'
        if not os.path.isdir(outcome_dir):
            os.makedirs(outcome_dir)

        # check if chunk has already been processed
        chunk = args['num'] - 1
        path = outcome_dir + '{}.pkl'.format(chunk)
        if os.path.isfile(path):
            print('Chunk {} already exists.'.format(chunk))
            exit(0)

        # process one chunk
        gen = gen_cls(**gen_args)
        df = gen.process_chunk(part=args['part'], chunk=chunk)

        # save
        topickle(df, path)

    else:
        # run in parallel on chunks
        sims = run_func_on_chunks(
            f=process_chunk_worker,
            func_kwargs=dict(
                part=args['part'],
                gen_class=gen_cls,
                gen_kwargs=gen_args
            )
        )

        # combine and process output
        process_sims(part=args['part'],
                     sims=sims,
                     output_dir=output_dir,
                     byr=args[BYR])


if __name__ == '__main__':
    main()
