# optimization parameters
LR_POLICY = 1e-4
LR_VALUE = 1e-3
RATIO_CLIP = 0.1
DROPOUT_POLICY = (.0, .0)
DROPOUT_VALUE = (.0, .1)
BATCH_SIZE = 4096

# state dictionaries
AGENT_STATE = 'agent_state_dict'
OPTIM_STATE = 'optimizer_state_dict'

# feat id
FEAT_TYPE = "feat_id"
NO_TIME = "no_time"
ALL_FEATS = "all"

PARAMS = {'byr': {'action': 'store_true'},      # buyer agent model
          'serial': {'action': 'store_true'},   # serial sampler
          'suffix': {'type': str},              # append to run id
          'norm': {'action': 'store_true'},     # normalize return
          'nocon': {'action': 'store_true'},    # accepts & rejects only
          'entropy': {'type': float, 'default': 1.}}

# count for stepping down entropy and stopping
PERIOD_EPOCHS = 1000
