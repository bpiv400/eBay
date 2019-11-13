import pandas as pd
import torch
import utils
from env_utils import get_model_input_paths, get_model_class, proper_squeeze


class ModelInterface:
    def __init__(self, composer):
        self.composer = composer

    @staticmethod
    def



