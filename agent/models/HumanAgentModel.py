import torch
from utils import load_sizes, load_state_dict
from nets.FeedForward import FeedForward
from constants import POLICY_SLR, POLICY_BYR


class HumanAgentModel(torch.nn.Module):
    """
    Predicts human behavior for calculating cross-entropy penalty.
    """
    def __init__(self, byr=None):
        super().__init__()

        # policy net
        name = POLICY_BYR if byr else POLICY_SLR
        sizes = load_sizes(name)
        self.net = FeedForward(sizes=sizes)

        # load pretrained model
        init_dict = load_state_dict(name)
        self.net.load_state_dict(init_dict, strict=True)

    def get_policy(self, input_dict=None):
        self.eval()

        # processing for single observations
        if input_dict['lstg'].dim() == 1:
            for elem_name, elem in input_dict.items():
                input_dict[elem_name] = elem.unsqueeze(0)

        logits = self.net(input_dict).squeeze()
        return logits
