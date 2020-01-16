from rlenv.env_utils import load_model, model_str
from featnames import CON, MSG, DELAY


class PlayerInterface:
    def __init__(self, composer=None, byr=False):
        # composer
        self.composer = composer

        # store names for each
        self.con_model_name = model_str(CON, byr=byr)
        self.msg_model_name = model_str(MSG, byr=byr)
        self.delay_model_name = model_str(DELAY, byr=byr)
        
        # load models
        self.msg_model = load_model(self.msg_model_name)
        self.con_model = load_model(self.con_model_name)
        self.delay_model = load_model(self.delay_model_name)


    def con(self, sources=None):
        x = self.composer.build_input_vector(self.con_model_name, sources=sources)
        theta = self.con_model(x)
        return theta


    def msg(self, sources=None):
        x = self.composer.build_input_vector(self.msg_model_name, sources=sources)
        theta = self.msg_model(x)
        return theta


    def delay(self, sources=None):
        x = self.composer.build_input_vector(self.delay_model_name, sources=sources)
        theta = self.delay_model(x)
        return theta
