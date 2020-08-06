import argparse
import os
import gc
import torch
from agent.RlTrainer import RlTrainer
from agent.const import PARAM_DICTS
from agent.eval.EvalGenerator import EvalGenerator
from agent.AgentModel import AgentModel
from rlenv.generate.util import process_sims
from utils import compose_args, set_gpu_workers, \
    run_func_on_chunks, process_chunk_worker
from constants import VALIDATION, NUM_RL_WORKERS, BYR


def simulate(part=None, byr=None, run_dir=None, env=None):
    # recreate model
    model = AgentModel(byr=byr)
    state_dict = torch.load(run_dir + 'params.pkl',
                            map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=True)

    # run in parallel on chunks
    sims = run_func_on_chunks(
        num_chunks=NUM_RL_WORKERS,
        f=process_chunk_worker,
        func_kwargs=dict(
            part=part,
            gen_class=EvalGenerator,
            gen_kwargs=dict(env=env, model=model)
        )
    )

    # combine and process output
    part_dir = run_dir + '{}/'.format(part)
    process_sims(part=part, sims=sims, output_dir=part_dir)


def startup():
    # command-line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--byr', action='store_true')
    for d in PARAM_DICTS.values():
        compose_args(arg_dict=d, parser=parser)
    args = vars(parser.parse_args())

    # set gpu and cpu affinity
    set_gpu_workers(gpu=args['gpu'], use_all=args['all'], spawn=True)

    # print to console
    for k, v in args.items():
        print('{}: {}'.format(k, v))

    # split parameters
    trainer_args = dict()
    for param_set, param_dict in PARAM_DICTS.items():
        trainer_args[param_set] = {k: args[k] for k in param_dict}
    trainer_args[BYR] = args[BYR]

    return trainer_args


def main():
    trainer_args = startup()
    trainer = RlTrainer(**trainer_args)

    # if logging and model has already been trained, quit
    if trainer_args['system']['log'] and os.path.isdir(trainer.run_dir):
        print('{} already exists.'.format(trainer.run_id))
        exit()

    # train, then clean up
    trainer.train()

    # simulate each listing once
    if trainer_args['system']['log']:
        simulate_args = dict(byr=trainer.byr,
                             run_dir=trainer.run_dir,
                             env=trainer.env)
        del trainer
        gc.collect()

        simulate(part=VALIDATION, **simulate_args)


if __name__ == '__main__':
    main()
