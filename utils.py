import argparse
import pickle
import psutil
from time import sleep
import torch
import torch.multiprocessing as mp
from torch.nn.functional import log_softmax
import numpy as np
from compress_pickle import load
from nets.FeedForward import FeedForward
from constants import DAY, MONTH, SPLIT_PCTS, INPUT_DIR, \
    MODEL_DIR, META_6, META_7, PARTITIONS, MODEL_PARTS_DIR, \
    MAX_DELAY_TURN, MAX_DELAY_ARRIVAL, NUM_CHUNKS, AGENT_PARTS_DIR
from featnames import DELAY, EXP, X_LSTG


def unpickle(file):
    """
    Unpickles a .pkl file encoded with Python 3
    :param file: str giving path to file
    :return: contents of file
    """
    return pickle.load(open(file, "rb"))


def get_remaining(lstg_start, delay_start):
    """
    Calculates number of delay intervals remaining in lstg.
    :param lstg_start: seconds from START to start of lstg.
    :param delay_start: seconds from START to beginning of delay window.
    """
    remaining = lstg_start - delay_start
    remaining += MAX_DELAY_ARRIVAL
    remaining /= MAX_DELAY_TURN
    remaining = np.minimum(1.0, remaining)
    return remaining


def extract_clock_feats(seconds):
    """
    Creates clock features from timestamps.
    :param seconds: seconds since START.
    :return: tuple of time_of_day sine transform and afternoon indicator.
    """
    sec_norm = (seconds % DAY) / DAY
    time_of_day = np.sin(sec_norm * np.pi)
    afternoon = sec_norm >= 0.5
    return time_of_day, afternoon


def is_split(con):
    """
    Boolean for whether concession is (close to) an even split.
    :param con: scalar or Series of concessions.
    :return: boolean or Series of booleans.
    """
    return con in SPLIT_PCTS


def get_months_since_lstg(lstg_start=None, time=None):
    """
    Float number of months between inputs.
    :param lstg_start: seconds from START to lstg start.
    :param time: seconds from START to focal event.
    :return: number of months between lstg_start and start.
    """
    return (time - lstg_start) / MONTH


def slr_norm(con=None, prev_byr_norm=None, prev_slr_norm=None):
    """
    Normalized offer for seller turn.
    :param con: current concession, between 0 and 1.
    :param prev_byr_norm: normalized concession from one turn ago.
    :param prev_slr_norm: normalized concession from two turns ago.
    :return: normalized distance of current offer from start_price to 0.
    """
    return 1 - con * prev_byr_norm - (1 - prev_slr_norm) * (1 - con)


def byr_norm(con=None, prev_byr_norm=None, prev_slr_norm=None):
    """
    Normalized offer for buyer turn.
    :param con: current concession, between 0 and 1.
    :param prev_byr_norm: normalized concession from two turns ago.
    :param prev_slr_norm: normalized concession from one turn ago.
    :return: normalized distance of current offer from 0 to start_price.
    """
    return (1 - prev_slr_norm) * con + prev_byr_norm * (1 - con)


def load_sizes(name):
    """
    Loads featnames dictionary for a model
    :param name: str giving name (e.g. hist, con_byr),
     see const.py for model names
    :return: dict
    """
    return load(INPUT_DIR + 'sizes/{}.pkl'.format(name))


def load_featnames(name):
    """
    Loads featnames dictionary for a model
    :param name: str giving name (e.g. hist, con_byr),
     see const.py for model names
    :return: dict
    """
    return load(INPUT_DIR + 'featnames/{}.pkl'.format(name))


def load_state_dict(name=None):
    """
    Loads state dict of a model
    :param name: string giving name of model (see consts)
    :return: dict
    """
    model_path = '{}{}.net'.format(MODEL_DIR, name)
    state_dict = torch.load(model_path,
                            map_location=torch.device('cpu'))
    return state_dict


def load_model(name, verbose=False, use_trained=True):
    """
    Initialize PyTorch network for some model
    :param str name: full name of the model
    :param verbose: boolean for printing statements
    :param use_trained: loads trained model when True
    :return: torch.nn.Module
    """
    if verbose:
        print('Loading {} model'.format(name))

    # create neural network
    sizes = load_sizes(name)
    net = FeedForward(sizes)  # type: torch.nn.Module

    if use_trained:
        # read in model parameters
        state_dict = load_state_dict(name=name)

        # load parameters into model
        net.load_state_dict(state_dict, strict=True)

    # eval mode
    for param in net.parameters(recurse=True):
        param.requires_grad = False
    net.eval()

    # use shared memory
    net.share_memory()

    return net


def run_func_on_chunks(f=None, func_kwargs=None):
    """
    Applies f to all chunks in parallel.
    :param f: function that takes chunk number as input along with
    other arguments
    :param func_kwargs: dictionary of other keyword arguments
    :return: list of worker-specific output
    """
    num_workers = min(NUM_CHUNKS, psutil.cpu_count() - 1)
    pool = mp.Pool(num_workers)
    jobs = []
    for i in range(NUM_CHUNKS):
        kw = func_kwargs.copy()
        kw['chunk'] = i
        jobs.append(pool.apply_async(f, kwds=kw))
    res = []
    for job in jobs:
        while True:
            if job.ready():
                res.append(job.get())
                break
            else:
                sleep(5)
    return res


def get_cut(meta):
    if meta in META_6:
        return .06
    if meta in META_7:
        return .07
    return .09


def get_model_predictions(m, x):
    """
    Returns predicted categorical distribution.
    :param str m: name of model
    :param dict x: dictionary of input tensors
    :return: torch tensor
    """
    # initialize neural net
    net = load_model(m, verbose=False)
    if torch.cuda.is_available():
        net = net.to('cuda')

    # split into batches
    vec = np.array(range(len(x['lstg'])))
    batches = np.array_split(vec, 1 + len(vec) // 2048)

    # convert to 32-bit numpy arrays
    if 'DataFrame' in str(type(x['lstg'])):
        x = {k: v.values.astype('float32') for k, v in x.items()}

    # model predictions
    p0 = []
    for b in batches:
        x_b = {k: torch.from_numpy(v[b, :]) for k, v in x.items()}
        if torch.cuda.is_available():
            x_b = {k: v.to('cuda') for k, v in x_b.items()}
        theta_b = net(x_b).cpu().double()
        p0.append(np.exp(log_softmax(theta_b, dim=-1)))

    # concatenate and return
    return torch.cat(p0, dim=0).numpy()


def input_partition():
    """
    Parses command line input for partition name.
    :return part: string partition name.
    """
    parser = argparse.ArgumentParser()

    # partition
    parser.add_argument('--part', required=True, type=str,
                        choices=PARTITIONS, help='partition name')
    return parser.parse_args().part


def init_optional_arg(kwargs=None, name=None, default=None):
    if name not in kwargs:
        kwargs[name] = default


def load_file(part, x, agent=None):
    """
    Loads file from partitions directory.
    :param str part: name of partition
    :param str x: name of file
    :param bool agent: use agent files if True
    :return: dataframe
    """
    folder = AGENT_PARTS_DIR if agent else MODEL_PARTS_DIR
    suffix = 'pkl' if x == X_LSTG else 'gz'
    return load('{}{}/{}.{}'.format(folder, part, x, suffix))


def drop_censored(df):
    """
    Removes censored observations from a dataframe of offers
    :param df: dataframe with index ['lstg', 'thread', 'index']
    :return: dataframe
    """
    censored = df[EXP] & (df[DELAY] < 1)
    return df[~censored]


def set_gpu_workers(gpu=None):
    """
    Sets the GPU index and the CPU affinity.
    """
    torch.cuda.set_device(gpu)
    print('Training on cuda:{}'.format(gpu))

    # set cpu affinity
    p = psutil.Process()
    start = NUM_CHUNKS * gpu
    workers = list(range(start, start + NUM_CHUNKS))
    p.cpu_affinity(workers)
    print('vCPUs: {}'.format(p.cpu_affinity()))


def compose_args(arg_dict=None, parser=None):
    for k, v in arg_dict.items():
        parser.add_argument('--{}'.format(k), **v)
