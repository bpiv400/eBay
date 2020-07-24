import argparse
import os
import gc
from agent.RlTrainer import RlTrainer
from agent.eval.EvalGenerator import EvalGenerator
from agent.util import get_values, save_run
from agent.const import PARAM_DICTS, KL_PENALTY
from rlenv.generate.util import process_sims
from utils import set_gpu_workers, compose_args, \
    unpickle, run_func_on_chunks, process_chunk_worker
from constants import BYR, DROPOUT, VALIDATION, MODEL_DIR, \
    POLICY_SLR, POLICY_BYR


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


def post_train(trainer=None, trainer_args=None):
    # # simulate outcomes
    # simulate(part=VALIDATION,
    #          run_dir=run_dir,
    #          model_kwargs=trainer_args['model_params'],
    #          agent_params=trainer_args['agent_params'])

    # # evaluate simulations
    # values = get_values(part=VALIDATION,
    #                     run_dir=run_dir,
    #                     prefs=prefs)

    # save run parameters
    save_run(log_dir=trainer.log_dir,
             run_id=trainer.run_id,
             econ_params=trainer_args['econ_params'],
             kl_penalty=trainer.algo.kl_penalty)


def main():
    # command-line parameters
    parser = argparse.ArgumentParser()
    for d in PARAM_DICTS.values():
        compose_args(arg_dict=d, parser=parser)
    args = vars(parser.parse_args())

    # set gpu and cpu affinity
    set_gpu_workers(gpu=args['gpu'], spawn=True)

    # add dropout
    s = unpickle(MODEL_DIR + 'dropout.pkl')
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
    base_run_id = trainer.run_id
    if args['log']:
        post_train(trainer=trainer, trainer_args=trainer_args)

    # put state dict in model parameters
    state_dict = trainer.agent.model.state_dict()
    trainer_args['model_params']['model_state_dict'] = state_dict

    # housekeeping
    del trainer
    gc.collect()

    # re-train with cross_entropy
    for i in range(len(KL_PENALTY)):
        print('Training with KL penalty: {}'.format(KL_PENALTY[i]))
        trainer = RlTrainer(kl_penalty_idx=i,
                            run_id=base_run_id,
                            **trainer_args)
        trainer.train()
        if args['log']:
            post_train(trainer=trainer, trainer_args=trainer_args)

        # housekeeping
        del trainer
        gc.collect()


if __name__ == '__main__':
    main()
