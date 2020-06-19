import psutil
from constants import BYR_PREFIX, SLR_PREFIX
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
AGENT_PARAMS = {'role': {'type': str,
                         'choices': [BYR_PREFIX, SLR_PREFIX],
                         'default': SLR_PREFIX},
                'delay': {'action': 'store_true'},
                'dropout0': {'type': float, 'default': 0.0},
                'dropout1': {'type': float, 'default': 0.0},
                'feat_id': {'type': str,
                            'choices': [ALL_FEATS, NO_TIME],
                            'default': ALL_FEATS},
                'con_type': {'type': str,
                             'choices': [FULL_CON, QUARTILES, HALF],
                             'default': FULL_CON},
                BYR_HIST: {'type': float,
                           'default': 0.5}
                }

BATCH_PARAMS = {'batch_size': {'type': int, 'default': 2 ** 9}}

PPO_PARAMS = {'minibatches': {'type': int, 'default': 1},
              'entropy_loss_coeff': {'type': float, 'default': .01},
              'cross_entropy': {'action': 'store_true'},
              'ratio_clip': {'type': float, 'default': .1},
              'clip_grad_norm': {'type': float, 'default': 1.},
              'patience': {'type': float, 'default': 1},
              'lr': {'type': float, 'default': .001},
              'same_lr': {'action': 'store_true'}}

SYSTEM_PARAMS = {'workers': {'type': int,
                             'default': psutil.cpu_count(logical=False)},
                 'auto': {'action': 'store_true'},
                 'multiple': {'action': 'store_true'},
                 'debug': {'action': 'store_true'},
                 'verbose': {'action': 'store_true'}}

PARAM_DICTS = {'agent_params': AGENT_PARAMS,
               'batch_params': BATCH_PARAMS,
               'ppo_params': PPO_PARAMS,
               'system_params': SYSTEM_PARAMS}

# list of parameters that must be shared among dictionaries
# tuple is of form (source dict, destination dict, [parameters])
DUPLICATE_PARAMS = [
    ('system_params', 'ppo_params', ['debug'])
]

# multi-processing parameters
THREADS_PER_PROC = 1

# final learning rate
LR1 = 1e-7