import pandas as pd
import torch
from torch.distributions.categorical import Categorical
import utils
from env_utils import get_model_dir, get_model_input_paths, get_model_class


class ModelInterface:
    def __init__(self, composer):
        self.composer = composer

    @staticmethod
    def _load_model(model_name, model_exp):
        """
        Initialize pytorch network for some model

        :param model_name: name of the model
        :param model_exp: experiment number for the model
        :return: PyTorch Module
        """
        err_name = model_name
        model_dir = get_model_dir(model_name)
        paths = get_model_input_paths(model_dir, model_exp)
        params_path, sizes_path, model_path = paths
        try:
            sizes = utils.unpickle(sizes_path)
            params = pd.read_csv(params_path, index_col='id')
            params = params.loc[model_exp].to_dict()
            model_class = get_model_class(model_name)
            net = model_class(params, sizes)
            net.load_state_dict(torch.load(model_path))
        except (RuntimeError, FileNotFoundError) as e:
            print(e)
            print('failed for {}'.format(err_name))
            return None
        return net

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
    def _categorical_sample(params, n):
        params = ModelInterface.proper_squeeze(params)
        cat = Categorical(logits=params)
        return cat.sample(sample_shape=(n, ))


