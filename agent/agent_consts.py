
# sub directory names
INPUT_DIR = "input"
TRAIN_SEED = 10


# seller input groups
seller_groupings = [
    'lstg',
    'w2v_byr',
    'w2v_slr',
    'cat',
    'cndtn',
]
for i in range(1, 7):
    seller_groupings.append('offer{}'.format(i))
    del i

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

# PPO hyperparameters
PPO_MINIBATCHES = 4
PPO_EPOCHS = 4


# SET THESE PARAMETERS
NUM_BATCHES = 100
BATCH_SIZE = 1000
BATCH_B = 8 # ENVS PER BATCH
BATCHES_PER_LOG = 2


# THESE ONES ARE DOWNSTREAM
TOTAL_STEPS = NUM_BATCHES * BATCH_SIZE
BATCH_T = int(BATCH_SIZE / BATCH_B)
LOG_INTERVAL_STEPS = BATCH_SIZE * BATCHES_PER_LOG


