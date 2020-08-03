from constants import POLICY_BYR
from utils import load_featnames

# optimization parameters
LR = 1e-4
RATIO_CLIP = 0.1
ENTROPY = [0.001]
KL = [0.]
DELTA = [.7, .8, .9, .995]

# state dictionaries
AGENT_STATE = 'agent_state_dict'
OPTIM_STATE = 'optimizer_state_dict'

# feat id
FEAT_TYPE = "feat_id"
NO_TIME = "no_time"
ALL_FEATS = "all"

PPO_PARAMS = {'delta': {'type': float, 'default': .995},
              'entropy_coeff': {'type': float, 'default': .001},
              'use_kl': {'action': 'store_true'}}

SYSTEM_PARAMS = {'gpu': {'type': int, 'default': 0},
                 'log': {'action': 'store_true'},
                 'exp': {'type': int},
                 'serial': {'action': 'store_true'},
                 'verbose': {'action': 'store_true'},
                 'all': {'action': 'store_true'},
                 'batch_size': {'type': int, 'default': 2 ** 12}}

PARAM_DICTS = {'ppo': PPO_PARAMS, 'system': SYSTEM_PARAMS}

# run for this many epochs
STOPPING_EPOCHS = 2500

# for identifying first turn in buyer model
T1_IDX = load_featnames(POLICY_BYR)['lstg'].index('t1')

# extra weight given to first-turn delay logit in byr model
DELAY_BOOST = 10
