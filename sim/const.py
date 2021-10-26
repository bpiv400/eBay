# simulator training hyperparameters
LR_FACTOR = 0.1  # multiply learning rate by this factor when training slows
LR0 = 1e-3  # initial learning rate
LR1 = 1e-7  # stop training when learning rate is lower than this
FTOL = 1e-2  # decrease learning rate when relative improvement in loss is less than this
AMSGRAD = True  # use AMSgrad version of ADAM if True
MBSIZE = {True: 128, False: 1e5}  # True for training, False for validation