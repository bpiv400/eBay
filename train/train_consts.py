# optimization parameters
NUM_WORKERS = {'first_arrival': 4,
               'hist': 7,
               'con1': 7,
               'init_byr': 5,
               'init_slr': 6,
               'listings': 6,
               'threads': 6,
               'threads_no_tf': 7}
MBSIZE = {True: 128, False: 2048}  # True for training, False for validation

# learning rate parameters
LR_FACTOR = 0.1  # multiply learning rate by this factor when training slows
LR0 = [1e-3]  # initial learning rates to search over
LR1 = 1e-8  # stop training when learning rate is lower than this
FTOL = 1e-2  # decrease learning rate when relative improvement in loss is less than this

# dropout grid
INT_DROPOUT = 10
MAX_DROPOUT = 7
DROPOUT_GRID = []
for i in range(0, MAX_DROPOUT):
    for j in range(i + 1, MAX_DROPOUT + 1):
        DROPOUT_GRID.append([i, j])

# type of normalization
NORM_TYPE = {'init_slr': 'weight', 'init_byr': 'weight'}
