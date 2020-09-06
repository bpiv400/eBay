# concession sets
FULL = 'full'
SPARSE = 'sparse'
NOCON = 'nocon'
CON_SETS = [FULL, SPARSE, NOCON]

# optimization parameters
LR_POLICY = 1e-4
LR_VALUE = 1e-3
RATIO_CLIP = 0.1
BATCH_SIZE = 4096
ENTROPY = {FULL: .01, SPARSE: .02, NOCON: .05}   # entropy bonus coefficient
ENTROPY_THRESHOLD = .05                          # stopping threshold
DEPTH_COEF = .1                                  # for incentivizing visits to later turns

# state dictionaries
AGENT_STATE = 'agent_state_dict'
OPTIM_STATE = 'optimizer_state_dict'

PARAMS = {'byr': {'action': 'store_true'},          # buyer agent model
          'dropout': {'type': int, 'default': 0},   # index for dropout
          'con_set': {'choices': CON_SETS,          # concession set
                      'default': FULL}}

# count for stepping down entropy and stopping
PERIOD_EPOCHS = 1000
