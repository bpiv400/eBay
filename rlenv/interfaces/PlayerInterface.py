from rlenv.env_utils import load_model, model_str
from featnames import CON, MSG, DELAY


class PlayerInterface:
    def __init__(self, composer=None, byr=False):
        #composer
        self.composer = composer

        #store names for each
        self.con_model_name = model_str(CON, byr=byr)
        self.msg_model_name = model_str(MSG, byr=byr)
        self.delay_model_name = model_str(DELAY, byr=byr)
        
        # load models
        self.msg_model = load_model(self.msg_model_name)
        self.con_model = load_model(self.con_model_name)
        self.delay_model = load_model(self.delay_model_name)

    def con(self, sources=None):
        input_dict = self.composer.build_input_vector(self.con_model_name, recurrent=False,
                                                      sources=sources, fixed=True)
        params = self.con_model(input_dict['x'])
        return params

    def msg(self, sources=None):
        input_dict = self.composer.build_input_vector(self.msg_model_name, recurrent=False,
                                                      sources=sources, fixed=True)
        params = self.msg_model(input_dict['x'])
        return params

    def delay(self, sources=None, hidden=None):
        input_dict = self.composer.build_input_vector(self.delay_model_name,
                                                      sources=sources, recurrent=True,
                                                      fixed=False)
        params, hidden = self.delay_model.step(x_time=input_dict['x_time'], hidden=hidden)
        return params, hidden

    def init_delay(self, sources=None):
        input_dict = self.composer.build_input_vector(self.delay_model_name, sources=sources,
                                                      recurrent=False, fixed=True)
        hidden = self.delay_model.init(x=input_dict['x'])
        return hidden