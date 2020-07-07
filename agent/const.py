from constants import HIST_QUANTILES
from featnames import BYR_HIST

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
AGENT_PARAMS = {BYR_HIST: {'type': int,
                           'choices': list(range(HIST_QUANTILES))},
                FEAT_TYPE: {'type': str,
                            'choices': [ALL_FEATS, NO_TIME],
                            'default': ALL_FEATS}}

MODEL_PARAMS = {'dropout_policy': {'type': tuple,
                                   'default': (0., 0.)},
                'dropout_value': {'type': tuple,
                                  'default': (0., 0.)}}

PPO_PARAMS = {'entropy_coeff': {'type': float, 'default': .01},
              'use_cross_entropy': {'action': 'store_true'},
              'ratio_clip': {'type': float, 'default': .1},
              'lr': {'type': float, 'default': .0002},
              'action_discount': {'type': float, 'default': 1.},
              'action_cost': {'type': float, 'default': 0.}}

SYSTEM_PARAMS = {'gpu': {'type': int, 'default': 1},
                 'exp': {'type': int},
                 'serial': {'action': 'store_true'},
                 'verbose': {'action': 'store_true'},
                 'batch_size': {'type': int, 'default': 2 ** 12}}

PARAM_DICTS = {'agent_params': AGENT_PARAMS,
               'model_params': MODEL_PARAMS,
               'ppo_params': PPO_PARAMS,
               'system_params': SYSTEM_PARAMS}

# multi-processing parameters
THREADS_PER_PROC = 1

# stopping parameters
STOPPING_WINDOW = 400
STOPPING_THRESHOLD = .001  # TODO: use .01
