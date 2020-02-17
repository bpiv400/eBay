
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
TOTAL_STEPS = 1000000000
NUM_BATCHES = 1000000
BATCH_T = TOTAL_STEPS / NUM_BATCHES
STEPS_PER_ENV = 100
BATCH_B = BATCH_T / STEPS_PER_ENV

# PPO hyperparameters
PPO_MINIBATCHES = 4
PPO_EPOCHS = 4
