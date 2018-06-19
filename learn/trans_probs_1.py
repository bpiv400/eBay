import pandas as np
import numpy as np
import sklearn as sk
import sys
import os


def load_data(counter):
    df = pd.read_csv('data/')


def main():
    # grab relevant arguments
    parser = argparse.ArgumentParser(
        description='associate threads with all relevant variables')
    parser.add_argument('--dir', action='store', type=str)
    parser = argparse.ArgumentParser(
        description='associate threads with all relevant variables')
    parser.add_argument('--count', action='store', type=str)
    subdir = args.dir

    # initialize model
    model = sk.neural_network.MLPClassifier
    # load data slice
    for i in range
    df = load_data(subdir, counter)


if __name__ == '__main__':
    main()
