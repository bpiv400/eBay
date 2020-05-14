from constants import FIRST_ARRIVAL_MODEL, SLR_POLICY_INIT, \
    DISCRIM_LISTINGS, DISCRIM_THREADS

# optimization parameters
NUM_WORKERS = {FIRST_ARRIVAL_MODEL: 4,
               SLR_POLICY_INIT: 6,
               DISCRIM_LISTINGS: 6,
               DISCRIM_THREADS: 6}
MBSIZE = {True: 128, False: 2048}  # True for training, False for validation

# learning rate parameters
LR_FACTOR = 0.1  # multiply learning rate by this factor when training slows
LR0 = [1e-3]  # initial learning rates to search over
LR1 = 1e-7  # stop training when learning rate is lower than this
FTOL = 1e-2  # decrease learning rate when relative improvement in loss is less than this

# dropout grid
INT_DROPOUT = 10
MAX_DROPOUT = 7
DROPOUT_GRID = []
for j in range(1, MAX_DROPOUT + 1):
    for i in range(0, j):
        if j - i <= 3:
            DROPOUT_GRID.append([i, j])
