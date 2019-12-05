import sys, os, argparse, math
import torch
import numpy as np, pandas as pd
from torch.utils.data import DataLoader
from simulator.interface import Sample, collateFF, collateRNN
from constants import *


def run_loop(simulator, data, optimizer=None):
    # training or validation
    isTraining = optimizer is not None

    # collate function
    f = collateRNN if data.isRecurrent else collateFF

    # sampler
    sampler = Sample(data, isTraining)

    # load batches
    batches = DataLoader(data, batch_sampler=sampler,
        collate_fn=f, num_workers=NUM_WORKERS, pin_memory=True)

    # loop over batches, calculate log-likelihood
    lnL = 0.0
    for batch in batches:
        lnL += simulator.run_batch(batch, optimizer)

    return lnL / data.N_labels