import psutil
from constants import BYR_PREFIX, SLR_PREFIX, DROPOUT

# state dictionaries
AGENT_STATE = 'agent_state_dict'
OPTIM_STATE = 'optimizer_state_dict'

# sub directory names
TRAIN_DIR = "train"
TRAIN_SEED = 10

# seller input groups
seller_groupings = [
    'lstg',
    'w2v_byr',
    'w2v_slr',
    'cat',
    'cndtn',
    'slr'
]
for i in range(1, 7):
    seller_groupings.append('offer{}'.format(i))
    del i

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
                             'default': FULL_CON}}

BATCH_PARAMS = {'batch_size': {'type': int, 'default': 2 ** 12}}

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

# multi-processing parameters
THREADS_PER_PROC = 1

# final learning rate
LR1 = 1e-7

# seconds in buyer arrival window
BUYER_ARRIVE_INTERVAL = 8 * 60 * 60
