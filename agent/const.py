from constants import DROPOUT_GRID
from featnames import BYR, CON_SET, DELTA, DROPOUT, ENTROPY

# concession sets
FULL = 'full'
SPARSE = 'sparse'
NOCON = 'nocon'

# optimization parameters
AGENT_HIDDEN = 512
LR_POLICY = 1e-4
LR_VALUE = 1e-3
RATIO_CLIP = 0.1
BATCH_SIZE = 4096
STOP_ENTROPY = .01

# state dictionaries
AGENT_STATE = 'agent_state_dict'

# agent parameters
DELTA_CHOICES = [.7, .9]
PARAMS = {BYR: dict(action='store_true'),
          CON_SET: dict(choices=[FULL, SPARSE, NOCON], default=FULL),
          DELTA: dict(type=float, choices=DELTA_CHOICES),
          ENTROPY: dict(type=float, default=.01),
          DROPOUT: dict(type=int, default=1,
                        choices=range(len(DROPOUT_GRID)))}

# epoch count for stepping down entropy and stopping
PERIOD_EPOCHS = {NOCON: 500, SPARSE: 1000, FULL: 1500}

# minimum byr concession on turn 1
BYR_MIN_CON1 = 50

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
