import numpy as np
from constants import DROPOUT_GRID, NUM_COMMON_CONS
from featnames import BYR, DELTA, DROPOUT, ENTROPY
from utils import load_feats

# concessions for agent to use
COMMON_CONS = load_feats('common_cons')
AGENT_CONS = COMMON_CONS.copy()
for t in range(1, 7):
    other = [0, 1] if t in [1, 3, 5] else [0, 1, 1.1]
    AGENT_CONS[t] = np.sort(np.concatenate([other, COMMON_CONS[t]]))
AGENT_CONS[7] = np.concatenate([np.zeros(NUM_COMMON_CONS + 1), [1.]])

# optimization parameters
LR_POLICY = 1e-4
LR_VALUE = 1e-3
RATIO_CLIP = 0.1
BATCH_SIZE = 4096
STOP_ENTROPY = .01

# state dictionaries
AGENT_STATE = 'agent_state_dict'

# agent parameters
DELTA_CHOICES = [0, .7, 1]
AGENT_PARAMS = {BYR: dict(action='store_true'),
                DELTA: dict(type=float,
                            choices=DELTA_CHOICES,
                            required=True)}

HYPER_PARAMS = {ENTROPY: dict(type=float, default=.01),
                DROPOUT: dict(type=int,
                              default=0,
                              choices=range(len(DROPOUT_GRID)))}

# epoch count for stepping down entropy and stopping
PERIOD_EPOCHS = 1000

# names for opt_info namedtuple
FIELDS = ["ActionsPerTraj", "ThreadsPerTraj", "DaysToDone",
          "Turn1_AccRate", "Turn1_RejRate", "Turn1_ConRate", "Turn1Con",
          "Turn2_AccRate", "Turn2_RejRate", "Turn2_ExpRate", "Turn2_ConRate", "Turn2Con",
          "Turn3_AccRate", "Turn3_RejRate", "Turn3_ConRate", "Turn3Con",
          "Turn4_AccRate", "Turn4_RejRate", "Turn4_ExpRate", "Turn4_ConRate", "Turn4Con",
          "Turn5_AccRate", "Turn5_RejRate", "Turn5_ConRate", "Turn5Con",
          "Turn6_AccRate", "Turn6_RejRate", "Turn6_ExpRate", "Turn6_ConRate", "Turn6Con",
          "Turn7_AccRate",
          "Rate_1", "Rate_2", "Rate_3", "Rate_4", "Rate_5", "Rate_6", "Rate_7", "Rate_Sale",
          "DollarReturn", "NormReturn", "Value", "Advantage", "Entropy",
          "Loss_Policy", "Loss_Value", "Loss_EntropyBonus"]
