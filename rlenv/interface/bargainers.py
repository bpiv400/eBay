from rlenv.interface.model_names import MSG, DELAY, CON
from rlenv.env_utils import load_model
from interface.model_names import model_str


class BargainerModel:
    def __init__(self, msg=0, con=0, delay=0, composer=None, byr=False):
        super(BargainerModel, self).__init__(composer)
        #store names for each
        self.con_model_name = model_str(CON, byr=byr)
        self.msg_model_name = model_str(MSG, byr=byr)
        self.delay_model_name = model_str(DELAY, byr=byr)
        # load models
        self.msg_model = load_model(self.msg_model_name, msg)
        self.con_model = load_model(self.con_model_name, con)
        self.delay_model = load_model(self.delay_model_name, delay)

    def con(self, sources=None, hidden=None, turn=0):
        fixed = hidden is None
        x_fixed, x_time = self.composer.build_input_vector(self.con_model_name,
                                                           sources=sources,
                                                           recurrent=True, fixed=fixed)
        params, hidden = self.con_model.simulate(x_time, x_fixed=x_fixed, hidden=hidden,
                                                 turn=turn)
        return params, hidden

    def msg(self, sources=None, hidden=None, turn=0):
        fixed = hidden is None
        x_fixed, x_time = self.composer.build_input_vector(self.msg_model_name,
                                                           sources=sources,
                                                           recurrent=True, fixed=fixed)
        params, hidden = self.msg_model.simulate(x_time, x_fixed=x_fixed, hidden=hidden,
                                                 turn=turn)
        return params, hidden

    def delay(self, sources=None, hidden=None, turn=0):
        fixed = hidden is None
        x_fixed, x_time = self.composer.build_input_vector(self.msg_model_name,
                                                           sources=sources,
                                                           recurrent=True, fixed=fixed)
        params, hidden = self.delay_model.simulate(x_time, x_fixed=x_fixed, hidden=hidden,
                                                   turn=turn)
        return params, hidden

