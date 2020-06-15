# optimization parameters
NUM_WORKERS = {'arrival': 3,
               'first_arrival': 5}
MBSIZE = {True: 128, False: 2048}  # True for training, False for validation

# learning rate parameters
LR_FACTOR = 0.1  # multiply learning rate by this factor when training slows
LR0 = [1e-3]  # initial learning rates to search over
LR1 = 1e-7  # stop training when learning rate is lower than this
FTOL = 1e-2  # decrease learning rate when relative improvement in loss is less than this

# dropout grid
INT_DROPOUT = 10
DROPOUT_GRID = []
for j in range(8):
    for i in range(j+1):
        if j - i <= 3:
            DROPOUT_GRID.append([i, j])
