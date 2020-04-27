from constants import REINFORCE_DIR, BYR_PREFIX, SLR_PREFIX

# threshold for likelihood of no arrivals
NO_ARRIVAL_CUTOFF = .50 ** (1.0 / 12)
NO_ARRIVAL = 'no_arrival'

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

BATCH_PARAMS = {'batch_size': {'type': int, 'default': 2 ** 13},
                'batch_count': {'type': int, 'default': 10}}

PPO_PARAMS = {'mbsize': {'type': int, 'default': 512},
              'epochs': {'type': int, 'default': 4},
              'entropy_loss_coeff': {'type': float, 'default': 1.0},
              'value_loss_coeff': {'type': float, 'default': 1.0},
              'ratio_clip': {'type': float, 'default': 0.1},
              'discount': {'type': float, 'default': 1.0},
              'gae_lambda': {'type': float, 'default': 1.0},
              'clip_grad_norm': {'type': float, 'default': 1.0},
              'learning_rate': {'type': float, 'default': 0.001}}

PARAM_DICTS = [AGENT_PARAMS, BATCH_PARAMS, PPO_PARAMS]

# multi-processing parameters
THREADS_PER_PROC = 1
