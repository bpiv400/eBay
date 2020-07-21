import argparse
import os
import pandas as pd
from agent.train import simulate
from agent.Prefs import SellerPrefs
from agent.const import ALL_FEATS, FEAT_TYPE
from constants import BYR, SLR, AGENT_DIR, TRAIN_RL, VALIDATION, DROPOUT
from featnames import BYR_HIST


def main():
    # command-line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, required=True)
    parser.add_argument('--run_id', type=str, required=True)
    args = parser.parse_args()
    exp, run_id = args.exp, args.run_id

    # directories
    log_dir = AGENT_DIR + '{}/{}/'.format(SLR, ALL_FEATS)
    run_dir = log_dir + 'run_{}/'.format(run_id)

    # simulate outcomes
    for part in [VALIDATION, TRAIN_RL]:
        part_dir = run_dir + '{}/'.format(part)
        if not os.path.isdir(part_dir):
            os.mkdir(part_dir)
        print('Simulating {}...'.format(part))
        simulate(part=part,
                 run_dir=run_dir,
                 model_kwargs={
                     BYR: False,
                     DROPOUT: (0., 0.)
                 },
                 agent_params={
                     BYR: False,
                     FEAT_TYPE: ALL_FEATS,
                     BYR_HIST: None
                 })

    # experiment parameters
    print('Using parameters for experiment {}'.format(exp))
    exp_path = AGENT_DIR + 'exps.csv'
    params = pd.read_csv(exp_path, index_col=0).loc[exp].to_dict()
    params['monthly_discount'] = .995
    params['action_discount'] = 1.
    params['action_cost'] = 0.
    params['cross_entropy'] = False
    params['batch_size'] = 4096

    # TODO: construct valuation from outcomes
    prefs = SellerPrefs(params=params)

    # save run parameters
    run_path = log_dir + 'runs.csv'
    if os.path.isfile(run_path):
        df = pd.read_csv(run_path, index_col=0)
    else:
        df = pd.DataFrame(index=pd.Index([], name='run_id'))
    for k, v in params.items():
        df.loc[run_id, k] = v
    df.to_csv(run_path)


if __name__ == '__main__':
    main()
