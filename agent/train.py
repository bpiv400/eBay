import argparse
import os
import gc
from compress_pickle import load
from agent.RlTrainer import RlTrainer
from agent.util import save_run
from agent.const import PARAM_DICTS
from agent.eval.EvalGenerator import EvalGenerator
from rlenv.generate.util import process_sims
from utils import set_gpu_workers, compose_args, \
    process_chunk_worker, run_func_on_chunks
from constants import BYR, DROPOUT, VALIDATION, \
    POLICY_BYR, POLICY_SLR, MODEL_DIR


def simulate(part=None, run_dir=None, agent_params=None, model_kwargs=None):
    # directory for simulation output
    part_dir = run_dir + '{}/'.format(part)
    if not os.path.isdir(part_dir):
        os.mkdir(part_dir)

    # arguments for generator
    eval_kwargs = dict(
        agent_params=agent_params,
        model_kwargs=model_kwargs,
        run_dir=run_dir,
        verbose=False
    )

    # run in parallel on chunks
    sims = run_func_on_chunks(
        f=process_chunk_worker,
        func_kwargs=dict(
            part=part,
            gen_class=EvalGenerator,
            gen_kwargs=eval_kwargs
        )
    )

    # combine and process output
    process_sims(part=part, parent_dir=run_dir, sims=sims)


def main():
    # command-line parameters
    parser = argparse.ArgumentParser()
    for d in PARAM_DICTS.values():
        compose_args(arg_dict=d, parser=parser)
    args = vars(parser.parse_args())

    # set gpu and cpu affinity
    set_gpu_workers(gpu=args['gpu'])

    # add dropout
    s = load(MODEL_DIR + 'dropout.pkl')
    args[DROPOUT] = s.loc[POLICY_BYR if args[BYR] else POLICY_SLR]

    # print to console
    for k, v in args.items():
        print('{}: {}'.format(k, v))

    # split parameters
    trainer_args = dict()
    for param_set, param_dict in PARAM_DICTS.items():
        trainer_args[param_set] = {k: args[k] for k in param_dict}

    # model parameters
    model_params = {BYR: args[BYR], DROPOUT: args[DROPOUT]}
    trainer_args['model_params'] = model_params

    # training with entropy bonus
    trainer = RlTrainer(**trainer_args)
    trainer.train()

    # extract path information and delete trainer
    log_dir = trainer.log_dir
    run_id = trainer.run_id
    del trainer
    gc.collect()

    # re-train with cross_entropy

    # when logging, simulate
    if args['log']:
        run_dir = log_dir + 'run_{}/'.format(run_id)

        # simulate outcomes
        simulate(part=VALIDATION,
                 run_dir=run_dir,
                 model_kwargs=trainer_args['model_params'],
                 agent_params=trainer_args['agent_params'])

        # TODO: evaluate simulations


        # save run parameters
        save_run(log_dir=log_dir,
                 run_id=run_id,
                 args=trainer_args['econ_params'])


if __name__ == '__main__':
    main()
