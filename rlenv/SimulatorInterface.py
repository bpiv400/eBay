import torch
from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.negative_binomial import NegativeBinomial
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
        TODO: Make pathing names function in utils or in parsing file

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
    def _beta_prep(params):
        """
        Reshapes the output of a model that parameterizes a mixed beta
        distribution to have shape (sample, k, 3) where k gives the number
        of components in the mixture:

        [i, j, 0] gives alpha for the jth component of the ith sample
        [i, j, 1] gives beta for the jth component of the ith sample
        [i, j, 2] gives the mixing coefficient logit for the jth component of the ith sample

        :param params: Same as input to _beta_sample
        :return:
        """
        sample_size = params.shape[0]
        params = params.reshape(sample_size, 3, -1).permute(0, 2, 1)
        params[:, :, [0, 1]] = params[:, :, [0, 1]] = torch.exp(params[:, :, [0, 1]]) + 1
        return params

    @staticmethod
    def _beta_ancestor(logits):
        """
        Makes the categorical distribution that governs the mixture for an
        arbitrary beta mixture

        :param logits: output of _beta_prep
        :return: torch.distributions.categorical.Categorical
        """
        ancestor = Categorical(logits=logits[:, :, 2])
        return ancestor

    @staticmethod
    def _mixed_beta_sample(params):
        """
        Samples a mixed beta distribution parameterized by the first index
        of the params tensor

        :param params: 2-dimensional tensor output by secs model or
        sliced from concession model. The first dimension separates the
        batch members and the second dimension gives parameters for each batch member.
        In the second dimension, the first k elements correspond to to alpha,
        the second k elements correspond to beta, and the third k correspond to mixing coefficients
        :return: 1-dimensional tensor containing 1 sample for each batch member's dist
        """
        # compute sample
        params = SimulatorInterface._beta_prep(params)
        ancestor = SimulatorInterface._beta_ancestor(params)
        draws = ancestor.sample(sample_shape=(1,))
        beta_params = params[torch.arange(params.shape[0]), draws[0, :], :]
        beta = Beta(beta_params[:, 0], beta_params[:, 1])
        sample = SimulatorInterface.proper_squeeze(beta.sample((1,)))
        return sample

    @staticmethod
    def _negative_binomial_sample(params):
        """
        Returns a sample from a batched negative binomial distribution

        :param params: n x 2  tensor containing model outputs that correspond
        to the parameters of a negative binomial distribution in each row
        :return: 1-dimensional tensor containing 1 sample drawn from each distribution
        """
        dist = NegativeBinomial(total_count=torch.exp(params[:, 0]), logits=params[:, 1])
        sample = SimulatorInterface.proper_squeeze(dist.sample((1, )))
        return sample

    def days(self, sources=None, hidden=None):
        """
        Returns the number of buyers who arrive on a particular day
        # TODO: Update after Etan makes feed forward

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
        sample = SimulatorInterface._negative_binomial_sample(params)
        return sample

    def loc(self, sources=None, num_byrs=None):
        """
        Returns indicators for whether each buyer in a set of size
        num_byrs is from the US
        :param sources:
        :param num_byrs: int number of buyers
        :return: 1d tensor giving whether each buyer is from the US
        """
        x_fixed, _ = self.composer.build_input_vector(model_name=LOC, sources=sources,
                                                      fixed=True, recurrent=False, size=1)
        params = self.models[LOC].simulate(x_fixed)
        locs = SimulatorInterface._bernoulli_sample(params, num_byrs)
        return locs

    def hist(self, sources=None, byr_us=None):
        """
        Returns the number of previous best offer threads each
        buyer has participated in

        :param sources: source vectors
        :param byr_us: 1d np.array giving whether each byr is from the us
        :return: 1d np.array giving number of experiences each buyer has had
        """
        us_count = torch.nonzero(byr_us).item()
        foreign = us_count < byr_us.shape[0]
        us = us_count > 0
        x_fixed = self.composer.build_hist_input(sources=sources, us=us, foreign=foreign)
        model_output = self.models[HIST].simulate(x_fixed)

        params = torch.zeros(byr_us.shape[0], model_output.shape[1])
        if foreign and us:
            foreign = byr_us == 0
            params[foreign, :] = model_output[0, :]
            params[~foreign, :] = model_output[1, :]
        else:
            params[:, :] = model_output[0, :]
        hists = SimulatorInterface._negative_binomial_sample(params)
        return hists

    def sec(self, sources=None, num_byrs=None):
        """
        Samples the time of day when the buyer arrives

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
