"""

"""
from compress_pickle import load
import torch
import pandas as pd
from rlenv.env_consts import REWARD_EXPERIMENT_PATH, SIM_COUNT
from rlenv.interface.model_names import DELAY, CON, MSG, BYR_HIST, NUM_OFFERS
from rlenv.interface.bargainers import *
from rlenv.interface.ArrivalInterface import ArrivalInterface
from rlenv.env_utils import model_str
from rlenv.rewards.RewardEnvironment import RewardEnvironment
from rlenv.composer.Composer import Composer


class RewardGenerator:
    """
    Attributes:
        exp_id: int giving the id of the current experiment
        params: dictionary containing parameters of the experiment:
            SIM_COUNT: number of times environment should simulator each lstg
            model in MODELS: integer giving the experiment of id for each model to be used in simulator
    """
    def __init__(self, path, exp_id):
        super(RewardGenerator, self).__init__()
        input_dict = load(path)
        self.x_lstg = input_dict['x_lstg']
        self.lookup = input_dict['lookup']
        self.exp_id = exp_id
        self.params = self._load_params()
        composer = Composer(self.params['composer'])
        self.buyer = BuyerModel(msg=self.params[model_str(MSG, byr=True)],
                                con=self.params[model_str(CON, byr=True)],
                                delay=self.params[model_str(DELAY, byr=True)],
                                composer=composer)
        self.seller = SellerModel(msg=self.params[model_str(MSG, byr=False)],
                                  con=self.params[model_str(CON, byr=False)],
                                  delay=self.params[model_str(DELAY, byr=False)],
                                  composer=composer)
        self.arrival = ArrivalInterface(byr_hist=self.params[BYR_HIST],
                                        num_offers=self.params[NUM_OFFERS],
                                        composer=composer)

    def _load_params(self):
        """
        Loads dictionary of parameters associated with the current experiment
        from experiments spreadsheet

        :return: dictionary containing parameter values
        """
        params = pd.read_csv(REWARD_EXPERIMENT_PATH)
        params.set_index('id', drop=True, inplace=True)
        params = params.loc[self.exp_id, :].to_dict()
        return params

    def generate(self):
        for lstg in self.x_lstg.index:
            x_lstg = torch.from_numpy(self.x_lstg.loc[lstg, :]).float()
            lookup = self.lookup.loc[lstg, :]
            environment = RewardEnvironment(buyer=self.buyer, seller=self.seller,
                                            arrival=self.arrival, x_lstg=x_lstg,
                                            lookup=lookup)
            for i in range(self.params[SIM_COUNT]):
                environment.reset()
                sale, price, time = environment.run()





