import numpy as np
import pandas as pd
from utils import unpickle
from paths import FEATS_DIR


# concessions for agent to use
COMMON_CONS = unpickle(FEATS_DIR + 'common_cons.pkl')
NUM_COMMON_CONS = len(COMMON_CONS[1])

AGENT_CONS = COMMON_CONS.copy()
for t in range(1, 7):
    other = [0, 1] if t in [1, 3, 5] else [0, 1, 1.1]
    AGENT_CONS[t] = np.sort(np.concatenate([other, COMMON_CONS[t]]))

# optimization parameters
LR = 1e-5               # learning rate
MAX_GRAD_NORM = .5      # for clipping gradients
BATCH_SIZE = 4096       # number of actions to collect per update
ENTROPY_COEF = .01      # initial weight on entropy on objective function
STOP_ENTROPY = .1       # stop when entropy reaches this value
PERIOD_EPOCHS = 1500    # epoch count for stepping down entropy

# economic parameters
DELTA_SLR = [0., .75]
DELTA_BYR = [.9, 1, 1.5, 2, 3]
TURN_COST_CHOICES = range(0, 5)

SLR_NAMES = {DELTA_SLR[0]: 'Impatient agent',
             DELTA_SLR[1]: 'Patient agent'}

# concessions for buyer heuristic
BYR_CONS = pd.DataFrame(columns=[1, 3, 5])
i = 0
for con1 in [.5, .6, .67, .75, .8]:
    for con3 in [.17, .2, .25, .33, .4, .5, 1.]:
        if con3 == 1:
            BYR_CONS.loc[i, :] = (con1, con3, -1)
            i += 1
        else:
            for con5 in [.17, .2, .25, .33, .4, .5, 1.]:
                BYR_CONS.loc[i, :] = (con1, con3, con5)
                i += 1


# names for opt_info namedtuple
FIELDS = ["ActionsPerTraj", "ThreadsPerTraj", "DaysToDone",
          "Turn1_AccRate", "Turn1_RejRate", "Turn1_ConRate", "Turn1Con",
          "Turn2_AccRate", "Turn2_RejRate", "Turn2_ExpRate", "Turn2_ConRate", "Turn2Con",
          "Turn3_AccRate", "Turn3_RejRate", "Turn3_ConRate", "Turn3Con",
          "Turn4_AccRate", "Turn4_RejRate", "Turn4_ExpRate", "Turn4_ConRate", "Turn4Con",
          "Turn5_AccRate", "Turn5_RejRate", "Turn5_ConRate", "Turn5Con",
          "Turn6_AccRate", "Turn6_RejRate", "Turn6_ExpRate", "Turn6_ConRate", "Turn6Con",
          "Rate_1", "Rate_2", "Rate_3", "Rate_4", "Rate_5", "Rate_6", "Rate_Sale",
          "DollarReturn", "NormReturn", "Value", "Advantage", "Entropy",
          "Loss_Policy", "Loss_GradNorm", "Loss_Value", "Loss_EntropyBonus"]
