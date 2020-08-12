# optimization parameters
LR_POLICY = 1e-4
LR_VALUE = 1e-3
RATIO_CLIP = 0.1
DROPOUT_POLICY = (.0, .0)
DROPOUT_VALUE = (.0, .1)

# state dictionaries
AGENT_STATE = 'agent_state_dict'
OPTIM_STATE = 'optimizer_state_dict'

# feat id
FEAT_TYPE = "feat_id"
NO_TIME = "no_time"
ALL_FEATS = "all"

PPO_PARAMS = {'entropy_coeff': {'type': float, 'default': .001}}

SYSTEM_PARAMS = {'gpu': {'type': int, 'default': 0},
                 'log': {'action': 'store_true'},
                 'exp': {'type': int},
                 'serial': {'action': 'store_true'},
                 'verbose': {'action': 'store_true'},
                 'all': {'action': 'store_true'},
                 'suffix': {'type': str},
                 'batch_size': {'type': int, 'default': 2 ** 12}}

PARAM_DICTS = {'ppo': PPO_PARAMS, 'system': SYSTEM_PARAMS}

# count for stepping down entropy and stopping
PERIOD_EPOCHS = 1000
