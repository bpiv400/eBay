import argparse
from datetime import datetime as dt
import torch
from compress_pickle import load
from agent.RlTrainer import RlTrainer
from agent.const import AGENT_STATE, PARAM_DICTS
from agent.util import save_params, compose_args
from utils import set_gpu_workers
from constants import MODEL_DIR


def main():
    set_gpu_workers()  # set gpu and cpu affinity

    # command-line parameters
    parser = argparse.ArgumentParser()
    for d in PARAM_DICTS.values():
        compose_args(arg_dict=d, parser=parser)
    args = vars(parser.parse_args())

    # add dropout
    s = load(MODEL_DIR + 'dropout.pkl')
    for name in ['policy', 'value']:
        args['dropout_{}'.format(name)] = \
            s.loc['{}_{}'.format(name, args['name'])]

    # print to console
    for k, v in args.items():
        print('{}: {}'.format(k, v))

    # split parameters
    trainer_args = dict()
    for param_set, param_dict in PARAM_DICTS.items():
        curr_params = dict()
        for k in param_dict.keys():
            curr_params[k] = args[k]
        trainer_args[param_set] = curr_params

    # initialize trainer
    trainer = RlTrainer(**trainer_args)

    # training loop
    t0 = dt.now()
    trainer.train()
    time_elapsed = (dt.now() - t0).total_seconds()

    if not trainer_args['system_params']['dev']:
        # save parameters to file
        save_params(run_id=trainer.run_id,
                    args=args,
                    time_elapsed=time_elapsed)

        # drop optimization parameters
        path = trainer.log_dir + 'run_{}/params.pkl'.format(
            trainer.run_id)
        d = torch.load(path)
        torch.save(d[AGENT_STATE], path)


if __name__ == '__main__':
    main()
