
# sub directory names
INPUT_DIR = "input"
TRAIN_SEED = 10


# hyperparameter name constants
# concession set constants
CON_TYPE = 'con_type'
FULL_CON = 'full'
QUARTILES = 'quart'
HALF = 'half'
# feat id
FEAT_ID = "feat_id"


# batch size hyperparameters
# for now, expecting no multiprocessing
TOTAL_STEPS = 1000
NUM_BATCHES = 10
BATCH_T = int(TOTAL_STEPS / NUM_BATCHES)
print("T: {}".format(BATCH_T))
STEPS_PER_ENV = 10
BATCH_B = int(BATCH_T / STEPS_PER_ENV)
print("B: {}".format(BATCH_B))
print(BATCH_B)

# PPO hyperparameters
PPO_MINIBATCHES = 4
PPO_EPOCHS = 4
