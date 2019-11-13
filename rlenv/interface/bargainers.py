from rlenv.interface.ModelInterface import ModelInterface
from constants import BYR_PREFIX, SLR_PREFIX
from rlenv.interface.model_names import MSG, DELAY, CON


class BargainerModel(ModelInterface):
    def __init__(self, msg=0, con=0, delay=0, composer=None, byr=False):
        super(BargainerModel, self).__init__(composer)
        if byr:
            model_type = BYR_PREFIX
        else:
            model_type = SLR_PREFIX
        # load models
        self.msg_model = ModelInterface._load_model(model_type, MSG, msg)
        self.con_model = ModelInterface._load_model(model_type, CON, con)
        self.delay_model = ModelInterface._load_model(model_type, DELAY, delay)
        #store names for each
        self.con_model_name = '{}_{}'.format(model_type, CON)
        self.msg_model_name = '{}_{}'.format(model_type, MSG)
        self.delay_model_name = '{}_{}'.format(model_type, DELAY)

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

