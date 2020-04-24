import argparse
import numpy as np
from agent.EvalGenerator import EvalGenerator
from agent.models import PgCategoricalAgentModel


def evaluate(i, model_path, verbose):
    eval_kwargs = {
        'composer': env_params_train['composer'],
        'model_kwargs': generate_model_kwargs(),
        'model_class': PgCategoricalAgentModel,
        'model_path': model_path,
        'record': False,
        'verbose': verbose
    }
    eval_generator = EvalGenerator(**eval_kwargs)
    rewards = eval_generator.process_chunk(i)
    return np.mean(rewards)


def main():
    # run_id and chunk number from command line

    # load chunk

    # loop over models


if __name__ == '__main__':
    main()
