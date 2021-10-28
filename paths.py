import os


DATA_DIR = '/data/eBay/'                  # where data is stored
PARTS_DIR = DATA_DIR + 'partitions/'      # post-partition features
SIM_DIR = DATA_DIR + 'sim/'               # simulated threads and offers
INDEX_DIR = DATA_DIR + 'index/'           # indices for input files
PCTILE_DIR = DATA_DIR + 'pctile/'         # percentiles of features
FEATS_DIR = DATA_DIR + 'feats/'           # pre-partion features
LOG_DIR = DATA_DIR + 'logs/'              # model logs
MODEL_DIR = DATA_DIR + 'models/'          # trained models
PLOT_DIR = DATA_DIR + 'plots/'            # for creating figures
INPUT_DIR = DATA_DIR + 'inputs/'          # inputs for models
SIZES_DIR = INPUT_DIR + 'sizes/'          # for initializing models
FEATNAMES_DIR = INPUT_DIR + 'featnames/'  # for testing
AGENT_DIR = DATA_DIR + 'agent/'           # contains agent training logs, models, and simulations
FIG_DIR = os.path.expanduser('~/Dropbox/eBay/figures/')  # for saving figures

folders = [elem for elem in globals().values()
           if type(elem) is str and elem.startswith(DATA_DIR)]
for folder in folders:
    if type(folder) is str and not os.path.isdir(folder):
        os.makedirs(folder)
