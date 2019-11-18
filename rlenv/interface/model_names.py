#TODO: Move all of this into env_consts

# arrival interface
from constants import SLR_PREFIX, BYR_PREFIX

NUM_OFFERS = 'arrival'
BYR_HIST = 'hist'

# offer interface
CON = 'con'
DELAY = 'delay'
MSG = 'msg'

# prefixes
ARRIVAL_PREFIX = 'arrival'

# model sets
FEED_FORWARD = [BYR_HIST]
ARRIVAL = [NUM_OFFERS, BYR_HIST]
RECURRENT = [NUM_OFFERS, CON, DELAY, MSG]


def model_str(model_name, byr=False):
    """
    returns the string giving the name of an offer model
    model (used to refer to the model in SimulatorInterface
     and Composer

    :param model_name: str giving base name
    :param byr: boolean indicating whether this is a buyer model
    :return:
    """
    if model_name in ARRIVAL:
        return model_name
    if not byr:
        name = '{}_{}'.format(model_name, SLR_PREFIX)
    else:
        name = '{}_{}'.format(model_name, BYR_PREFIX)
    return name


LSTM_MODELS = [NUM_OFFERS, model_str(DELAY, byr=False), model_str(DELAY, byr=True)]
OFFER_NO_PREFIXES = [model for model in RECURRENT if model != NUM_OFFERS]
MODELS_NO_PREFIXES = RECURRENT + FEED_FORWARD
OFFER = [model_str(model, byr=False) for model in OFFER_NO_PREFIXES] + \
        [model_str(model, byr=True) for model in OFFER_NO_PREFIXES]
MODELS = OFFER + ARRIVAL

