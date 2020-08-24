from utils import load_featnames
from rlenv.const import LSTG_MAP
from constants import FIRST_ARRIVAL_MODEL

# optimization parameters
LR_POLICY = 1e-4
LR_VALUE = 1e-3
RATIO_CLIP = 0.1
DROPOUT_POLICY = (.0, .0)
DROPOUT_VALUE = (.0, .1)
BATCH_SIZE = 4096
ENTROPY = .01

# state dictionaries
AGENT_STATE = 'agent_state_dict'
OPTIM_STATE = 'optimizer_state_dict'

PARAMS = {'byr': {'action': 'store_true'},      # buyer agent model
          'serial': {'action': 'store_true'},   # serial sampler
          'name': {'type': str},                # run id
          'nocon': {'action': 'store_true'}}    # accepts & rejects only

# count for stepping down entropy and stopping
PERIOD_EPOCHS = 1000

# indices of lstg features to remove from buyer agent model
to_remove = ['lstg_ct', 'bo_ct',
             'auto_accept', 'auto_decline',
             'has_accept', 'has_decline']
x_lstg_feats = load_featnames(FIRST_ARRIVAL_MODEL)[LSTG_MAP]
BYR_DROP_IND = [x_lstg_feats.index(c) for c in to_remove]
