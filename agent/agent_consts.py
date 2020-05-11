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
                'delay': {'type': bool, 'default': False},
                DROPOUT: {'nargs': 2,
                          'type': float,
                          'default': [0.0, 0.0]
                          },
                'feat_id': {'type': str,
                            'choices': [ALL_FEATS, NO_TIME],
                            'default': ALL_FEATS},
                'con_type': {'type': str,
                             'choices': [FULL_CON, QUARTILES, HALF],
                             'default': FULL_CON}}

BATCH_PARAMS = {'batch_size': {'type': int, 'default': 2 ** 12}}

PPO_PARAMS = {'mbsize': {'type': int, 'default': 512},
              'cross_entropy_loss_coeff': {'type': float, 'default': 1.},
              'entropy_loss_coeff': {'type': float, 'default': 1.},
              'ratio_clip': {'type': float, 'default': .1},
              'clip_grad_norm': {'type': float, 'default': 1.},
              'patience': {'type': float, 'default': 1},
              'lr_policy': {'type': float, 'default': .001},
              'lr_value': {'type': float, 'default': .001}}

PARAM_DICTS = [AGENT_PARAMS, BATCH_PARAMS, PPO_PARAMS]

# multi-processing parameters
THREADS_PER_PROC = 1

# final learning rate
LR1 = 1e-7
