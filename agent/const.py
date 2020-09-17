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
DROPOUT = (.0, .1)
ENTROPY = {FULL: .02, SPARSE: .04, NOCON: .1}    # entropy bonus coefficient
ENTROPY_THRESHOLD = .01                          # stopping threshold

# state dictionaries
AGENT_STATE = 'agent_state_dict'
OPTIM_STATE = 'optimizer_state_dict'

PARAMS = {'byr': {'action': 'store_true'},          # buyer agent model
          'con_set': {'choices': CON_SETS,          # concession set
                      'default': FULL}}

# count for stepping down entropy and stopping
PERIOD_EPOCHS = 1000

# for constructing buyer values
DISCOUNT_RATE = .9
