import numpy as np
from constants import NUM_COMMON_CONS
from utils import load_feats


# concessions for agent to use
COMMON_CONS = load_feats('common_cons')
AGENT_CONS = COMMON_CONS.copy()
for t in range(1, 7):
    other = [0, 1] if t in [1, 3, 5] else [0, 1, 1.1]
    AGENT_CONS[t] = np.sort(np.concatenate([other, COMMON_CONS[t]]))
AGENT_CONS[7] = np.concatenate([np.zeros(NUM_COMMON_CONS + 1), [1.]])

# optimization parameters
LR_POLICY = 1e-4        # learning rate for policy network
LR_VALUE = 1e-3         # learning rate for value network
RATIO_CLIP = 0.1        # for PPO
BATCH_SIZE = 4096       # number of actions to collect per update
ENTROPY_COEF = .01      # initial weight on entropy on objective function
STOP_ENTROPY = .1       # stop when entropy reaches this value
PERIOD_EPOCHS = 1500    # epoch count for stepping down entropy

# state dictionaries
AGENT_STATE = 'agent_state_dict'

# economic parameters
DELTA_CHOICES = [0., .7]

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
