from constants import REINFORCE_DIR, BYR_PREFIX, SLR_PREFIX

# files
SELLER_TRAIN_INPUT = REINFORCE_DIR + 'train/seller.hdf5'

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

PARAM_SHARING = True

# command-line parameters
AGENT_PARAMS = {'role': {'type': str,
                         'choices': [BYR_PREFIX, SLR_PREFIX],
                         'default': SLR_PREFIX},
                'delay': {'type': bool, 'default': False},
                'feat_id': {'type': str,
                            'choices': [ALL_FEATS, NO_TIME],
                            'default': ALL_FEATS},
                'con_type': {'type': str,
                             'choices': [FULL_CON, QUARTILES, HALF],
                             'default': FULL_CON}}

BATCH_PARAMS = {'batch_size': {'type': int, 'default': 2 ** 12},
                'batch_count': {'type': int, 'default': 10}}

PPO_PARAMS = {'mbsize': {'type': int, 'default': 512},
              'cross_entropy_loss_coeff': {'type': float, 'default': 1.},
              'entropy_loss_coeff': {'type': float, 'default': 1.},
              'value_loss_coeff': {'type': float, 'default': 1.},
              'ratio_clip': {'type': float, 'default': .1},
              'clip_grad_norm': {'type': float, 'default': 1.},
              'learning_rate': {'type': float, 'default': .001}}

PARAM_DICTS = [AGENT_PARAMS, BATCH_PARAMS, PPO_PARAMS]

# multi-processing parameters
THREADS_PER_PROC = 1
