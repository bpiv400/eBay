# concession sets
FULL = 'full'
SPARSE = 'sparse'
NONE = 'none'
CON_SETS = [FULL, SPARSE, NONE]

# optimization parameters
LR_POLICY = 1e-4
LR_VALUE = 1e-3
RATIO_CLIP = 0.1
DROPOUT = (.0, .1)
BATCH_SIZE = 4096
ENTROPY = {FULL: .002, SPARSE: .004, NONE: .01}

# state dictionaries
AGENT_STATE = 'agent_state_dict'
OPTIM_STATE = 'optimizer_state_dict'

PARAMS = {'byr': {'action': 'store_true'},      # buyer agent model
          'serial': {'action': 'store_true'},   # serial sampler
          'name': {'type': str},                # run id
          'con_set': {'choices': CON_SETS,      # concession set
                      'default': FULL}}

# count for stepping down entropy and stopping
PERIOD_EPOCHS = 1000
