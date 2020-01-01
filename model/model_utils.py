from model.FeedForwardDataset import FeedForwardDataset
from model.RecurrentDataset import ArrivalDataset, DelayDataset


def get_dataset(name):
    if name == 'arrival':
        return ArrivalDataset

    if 'delay' in name:
        return DelayDataset
    
    return FeedForwardDataset