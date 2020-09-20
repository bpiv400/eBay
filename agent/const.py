# concession sets
FULL = 'full'
SPARSE = 'sparse'
NOCON = 'nocon'
CON_SETS = [FULL, SPARSE, NOCON]

# optimization parameters
LR_POLICY = 1e-4
LR_VALUE = 1e-3
RATIO_CLIP = 0.1
BATCH_SIZE = 8192
DROPOUT = (.0, .1)
# ENTROPY = {FULL: .02, SPARSE: .04, NOCON: .1}    # entropy bonus coefficient
ENTROPY = {FULL: .004, SPARSE: .008, NOCON: .02}
ENTROPY_THRESHOLD = .02                          # stopping threshold
LSTG_SIM_CT = 32                                 # number of times to simulate each lstg

# state dictionaries
AGENT_STATE = 'agent_state_dict'
OPTIM_STATE = 'optimizer_state_dict'

PARAMS = {'byr': {'action': 'store_true'},          # buyer agent model
          'con_set': {'choices': CON_SETS,          # concession set
                      'default': FULL}}

# count for stepping down entropy and stopping
PERIOD_EPOCHS = 750

# for constructing buyer values
DISCOUNT_RATE = .9
