import os
from agent.eval.AgentGenerator import AgentGenerator
from agent.eval.SlrRejectEnv import SlrRejectEnv
from agent.models.AgentModel import load_agent_model
from agent.models.HeuristicByr import HeuristicByr
from agent.models.HeuristicSlr import HeuristicSlr
from agent.eval.util import sim_run_dir, sim_args
from rlenv.generate.Generator import OutcomeGenerator
from rlenv.generate.util import process_sims
from utils import topickle, run_func_on_chunks, process_chunk_worker
from featnames import BYR, DELTA


def main():
    args = sim_args(num=True)

    # run directory
    run_dir = sim_run_dir(args)

    # generator
    if args['slrrej']:
        gen_cls = OutcomeGenerator
        gen_args = dict(env=SlrRejectEnv)
    else:
        gen_cls = AgentGenerator
        if args['heuristic']:
            model_cls = HeuristicByr if args[BYR] else HeuristicSlr
            model = model_cls(args[DELTA])
        else:
            model_args = {BYR: args[BYR], 'value': False}
            model = load_agent_model(model_args=model_args, run_dir=run_dir)
        gen_args = dict(model=model, byr=args[BYR], slr=not args[BYR])

    # generate
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
        output_dir = run_dir + '{}/'.format(args['part'])
        process_sims(part=args['part'],
                     sims=sims,
                     output_dir=output_dir,
                     byr=args[BYR])


if __name__ == '__main__':
    main()
