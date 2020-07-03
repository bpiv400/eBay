from featnames import BYR_HIST

# state dictionaries
AGENT_STATE = 'agent_state_dict'
OPTIM_STATE = 'optimizer_state_dict'

# sub directory names
TRAIN_DIR = "train"
TRAIN_SEED = 10

# concession set constants
CON_TYPE = 'con_type'
FULL_CON = 'full'
QUARTILES = 'quartiles'
HALF = 'half'

# feat id
FEAT_TYPE = "feat_id"
NO_TIME = "no_time"
ALL_FEATS = "all"

# command-line parameters
AGENT_PARAMS = {'name': {'type': str,
                         'choices': ['slr', 'slr_delay', 'byr_delay'],
                         'default': 'slr'},
                'feat_id': {'type': str,
                            'choices': [ALL_FEATS, NO_TIME],
                            'default': ALL_FEATS},
                'con_type': {'type': str,
                             'choices': [FULL_CON, QUARTILES, HALF],
                             'default': FULL_CON},
                BYR_HIST: {'type': float, 'default': 0.5}
                }

MODEL_PARAMS = {'dropout_policy': {'type': tuple,
                                   'default': (0., 0.)},
                'dropout_value': {'type': tuple,
                                  'default': (0., 0.)},
                'untrained': {'action': 'store_true'}}

PPO_PARAMS = {'entropy_coeff': {'type': float, 'default': .01},
              'use_cross_entropy': {'action': 'store_true'},
              'ratio_clip': {'type': float, 'default': .1},
              'clip_grad_norm': {'type': float, 'default': 1.},
              'lr': {'type': float, 'default': .001},
              'same_lr': {'action': 'store_true'},
              'action_discount': {'type': float, 'default': 1.},
              'action_cost': {'type': float, 'default': 0.},
              'debug_ppo': {'action': 'store_true'}}

SYSTEM_PARAMS = {'serial': {'action': 'store_true'},
                 'no_logging': {'action': 'store_true'},
                 'verbose': {'action': 'store_true'},
                 'batch_size': {'type': int, 'default': 2 ** 12}}

PARAM_DICTS = {'agent_params': AGENT_PARAMS,
               'model_params': MODEL_PARAMS,
               'ppo_params': PPO_PARAMS,
               'system_params': SYSTEM_PARAMS}

# multi-processing parameters
THREADS_PER_PROC = 1
