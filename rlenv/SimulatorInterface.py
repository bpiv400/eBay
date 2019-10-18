import torch
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.poisson import Poisson
import pandas as pd
import utils
from rlenv.model_names import *
from rlenv.Composer import Composer
from rlenv import env_consts
from simulator.nets import FeedForward, RNN, LSTM
from constants import TOL_HALF, SLR_PREFIX, BYR_PREFIX


class SimulatorInterface:
    def __init__(self, params):
        """
        Use rl experiment params to initialize models

        :param params:
        """
        self.models = dict()
        self.params = params
        for model in MODELS:
            self.models[model] = self._init_model(model, params[model])
        self.composer = Composer(params)

    @staticmethod
    def _init_model(model_name, model_exp):
        """
        Initialize pytorch network for some model
        TODO: Will need to update to accomodate new days and secs models

        :param model_exp: experiment number for the model
        :return: PyTorch Module
        """
        err_name = model_name
        model_dir = SimulatorInterface._model_dir(model_name)
        paths = SimulatorInterface._input_paths(model_dir, model_exp)
        params_path, sizes_path, model_path = paths
        try:
            sizes = utils.unpickle(sizes_path)
            params = pd.read_csv(params_path, index_col='id')
            params = params.loc[model_exp].to_dict()
            model_type = SimulatorInterface._model_type(model_name)
            net = model_type(params, sizes)
            net.load_state_dict(torch.load(model_path))
        except (RuntimeError, FileNotFoundError) as e:
            print(e)
            print('failed for {}'.format(err_name))
            return None
        return net

    @staticmethod
    def _model_type(model_name):
        """
        Returns the class of the given model
        TODO: Update to accommodate new days and secs models

        :param model_name: str giving the name of the model
        :return: simulator.nets.RNN, simulator.nets.LSTM, or
        simulator.nets.FeedForward
        """
        if model_name in FEED_FORWARD:
            mod_type = FeedForward
        elif model_name in LSTM_MODELS:
            mod_type = LSTM
        else:
            mod_type = RNN
        return mod_type

    @staticmethod
    def _input_paths(model_dir, exp):
        """
        Helper method that returns the paths to files related to some model, given
        that model's path and experiment number

        :param model_dir: string giving path to model directory
        :param exp: int giving integer number of the experiment
        :return: 3-tuple of params path, sizes path, model path
        """
        params_path = '{}params.csv'.format(model_dir)
        sizes_path = '{}sizes.pkl'.format(model_dir)
        model_path = '{}{}.pt'.format(model_dir, exp)
        return params_path, sizes_path, model_path

    @staticmethod
    def _model_dir(model_name):
        """
        Helper method that returns the path to a model's directory, given
        the name of that model
        TODO: Update to accommodate new days and secs models

        :param model_name: str naming model (following naming conventions in rlenv/model_names.py)
        :return: str
        """
        # get pathing names
        if SLR_PREFIX in model_name:
            model_type = SLR_PREFIX
            model_name = model_name.replace('{}_'.format(SLR_PREFIX), '')
        elif BYR_PREFIX in model_name:
            model_type = BYR_PREFIX
            model_name = model_name.replace('{}_'.format(BYR_PREFIX), '')
        else:
            model_type = ARRIVAL_PREFIX

        model_dir = '{}/{}/{}/'.format(env_consts.MODEL_DIR,
                                       model_type, model_name)
        return model_dir

    @staticmethod
    def proper_squeeze(tensor):
        """
        Squeezes a tensor to 1 rather than 0 dimensions

        :param tensor: torch.tensor with only 1 non-singleton dimension
        :return: 1 dimensional tensor
        """
        tensor = tensor.squeeze()
        if len(tensor.shape) == 0:
            tensor = tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _bernoulli_sample(logits, sample_size):
        """
        Returns sample of bernoulli distributions defined by logits

        :param logits: 2 dimensional tensor containing logits for a batch of
        distributions. Currently, expects 1 distribution
        :param sample_size: number of samples to draw from each distribution
        :return: 1-dimensional tensor containing
        """
        dist = Bernoulli(logits=logits)
        return SimulatorInterface.proper_squeeze(dist.sample((sample_size, )))

    @staticmethod
    def _poisson_sample(params):
        """
        Draws a sample from a poisson distribution parameterized by
        the input tensor according to documentation in ebay/documents

        :param params: 1-dimensional torch.tensor output by a poisson model
        :return: torch.LongTensor containing 1 element drawn from poisson
        """
        params = torch.exp(params)
        dist = Poisson(params)
        sample = dist.sample()
        return sample

    def days(self, sources=None, hidden=None):
        """
        Returns the number of buyers who arrive on a particular day
        # TODO: Check after Etan makes feed forward and poisson

        :param sources: dictionary containing entries for all  input maps (see env_consts)
        required to construct the model's inputs from source vectors in the environment
        :param hidden: tensor giving the hidden state of the model up to this point
        """
        fixed = hidden is None
        x_fixed, x_time = self.composer.build_input_vector(DAYS, sources=sources,
                                                           fixed=fixed, recurrent=True,
                                                           size=1)
        params, hidden = self.models[DAYS].simulate(x_time, x_fixed=x_fixed, hidden=hidden)
        params = params.squeeze().unsqueeze(0)
        sample = SimulatorInterface._poisson_sample(params)
        return sample

    def sec(self, sources=None, num_byrs=None):
        """
        Samples the time of day when the buyer arrives
        # TODO: Update after Etan replaces sec model with KDE
        :param sources: dictionary containing entries for all  input maps (see env_consts)
        required to construct the model's inputs from source vectors in the environment
        :param num_byrs: total number of buyers
        :return: a tensor of length num_byrs where each element contains a float in [0, 1],
        giving the time of day the buyer arrives in
        """
        x_fixed, _ = self.composer.build_input_vector(SEC, sources=sources, recurrent=False,
                                                      size=num_byrs, fixed=True)
        params = self.models[SEC].simulate(x_fixed)
        times = SimulatorInterface._mixed_beta_sample(params)
        return times

    def bin(self, sources=None, num_byrs=None):
        """
        Runs the bin model to determine whether each buyer chooses to buy the listing now
        upon arrival

        # TODO: ensure sources includes num_byr length vectors for byr_us and byr_hist

        :param sources: dictionary containing entries for all  input maps (see env_consts)
        required to construct the model's inputs from source vectors in the environment
        :param num_byrs: total number of buyers
        :return: tensor containing a {0, 1} bin indicator for each buyer
        """
        x_fixed, _ = self.composer.build_input_vector(BIN, sources=sources, recurrent=False,
                                                      size=num_byrs, fixed=True)
        params = self.models[BIN].simulate(x_fixed)
        bins = SimulatorInterface._bernoulli_sample(params, 1)
        return bins

    def cn(self, sources=None, hidden=None, model_name=None, sample=True):
        """
        Samples an offer from the relevant concession model and returns
        the concession value along with the total normalized concession and an indicator for
        split

        :param sources: dictionary containing entries for all  input maps (see env_consts)
        required to construct the model's inputs from source vectors in the environment
        :param hidden: tensor giving the hidden state of the model up to this point
        :param model_name: string giving name of model
        :param sample: boolean giving whether a sample should be drawn from the parameterized dist
        :return: 4-tuple containing an a float [0, 1] drawn from the distribution parameters
        the model outputs, the corresponding normalized concession value, an indicator
        for split, and the hidden state after processing  the turn
        """
        fixed = hidden is None
        x_fixed, x_time = self.composer.build_input_vector(model_name, sources=sources,
                                                           recurrent=True, size=1, fixed=fixed)
        params, hidden = self.models[model_name].simulate(x_time, x_fixed=x_fixed, hidden=hidden)
        if not sample:
            return 0, 0, 0, hidden
        cn = SimulatorInterface._mixed_beta_sample(params[0, :, :])
        # compute norm, split, and cn
        split = 1 if abs(.5 - cn) < TOL_HALF else 0
        # slr norm
        if SLR_PREFIX in model_name:
            norm = 1 - cn * sources[env_consts.O_OUTCOMES_MAP][2] - \
                   (1 - sources[env_consts.L_OUTCOMES_MAP][2]) * (1 - cn)
        # byr norm
        else:
            norm = (1 - sources[env_consts.O_OUTCOMES_MAP][2]) * cn + \
                sources[env_consts.L_OUTCOMES_MAP][2] * (1 - cn)
        return cn, norm, split, hidden

    def offer_indicator(self, model_name, sources=None, hidden=None, sample=True):
        """
        Computes outputs of recurrent offer models that produce an indicator for some event
        (round, msg, nines, accept, reject, delay)

        :param model_name: str giving name of the target model (see model_names.MODELS for
        valid model names)
        :param sources: dictionary containing entries for all  input maps (see env_consts)
        required to construct the model's inputs from source vectors in the environment
        :param hidden: tensor giving the hidden state of the model up to this point
        :param sample: boolean indicating whether a sample should be drawn
        :return: 2-tuple containing an int {0, 1} drawn from the model's parameters and
        a tensor giving the hidden state after this time step
        """
        fixed = hidden is None
        x_fixed, x_time = self.composer.build_input_vector(model_name, sources=sources,
                                                           recurrent=True, size=1, fixed=True)
        params, hidden = self.models[model_name].simulate(x_time, x_fixed=x_fixed, hidden=hidden)
        samp = 0
        if sample:
            samp = SimulatorInterface._bernoulli_sample(params[0, :, :], 1)
        return samp, hidden
