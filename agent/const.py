from constants import HIST_QUANTILES, POLICY_BYR, BYR
from featnames import BYR_HIST
from utils import load_featnames

# state dictionaries
AGENT_STATE = 'agent_state_dict'
OPTIM_STATE = 'optimizer_state_dict'

# sub directory names
TRAIN_DIR = "train"
TRAIN_SEED = 10

# feat id
FEAT_TYPE = "feat_id"
NO_TIME = "no_time"
ALL_FEATS = "all"

# command-line parameters
AGENT_PARAMS = {BYR: {'action': 'store_true'},
                FEAT_TYPE: {'type': str,
                            'choices': [ALL_FEATS, NO_TIME],
                            'default': ALL_FEATS},
                BYR_HIST: {'type': int,
                           'default': 5,
                           'choices': list(range(HIST_QUANTILES))}}

ECON_PARAMS = {'monthly_discount': {'type': float, 'default': .995},
               'action_discount': {'type': float, 'default': 1.},
               'action_cost': {'type': float, 'default': 0.}}

PPO_PARAMS = {'entropy_coeff': {'type': float, 'default': .01},
              'use_cross_entropy': {'action': 'store_true'},
              'ratio_clip': {'type': float, 'default': .1},
              'lr': {'type': float, 'default': .0002}}

SYSTEM_PARAMS = {'gpu': {'type': int, 'default': 0},
                 'log': {'action': 'store_true'},
                 'exp': {'type': int},
                 'serial': {'action': 'store_true'},
                 'verbose': {'action': 'store_true'},
                 'batch_size': {'type': int, 'default': 2 ** 12}}

PARAM_DICTS = {'agent_params': AGENT_PARAMS,
               'econ_params': ECON_PARAMS,
               'ppo_params': PPO_PARAMS,
               'system_params': SYSTEM_PARAMS}

# run for this many epochs
STOPPING_EPOCHS = 5

# for identifying first turn in buyer model
T1_IDX = load_featnames(POLICY_BYR)['lstg'].index('t1')

# extra weight given to first-turn delay logit in byr model
DELAY_BOOST = 10
