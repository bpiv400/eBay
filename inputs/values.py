from inputs.util import save_files, get_ind_x
from agent.util import load_values
from utils import load_file, input_partition
from featnames import LOOKUP, VALUES_MODEL

DELTA = .9


def process_inputs(part):
    # values
    y = load_values(part=part, delta=DELTA)

    # indices for listing features
    lstgs = load_file(part, LOOKUP).index
    idx_x = get_ind_x(lstgs=lstgs, idx=y.index)

    return {'y': y, 'idx_x': idx_x}


def main():
    # extract parameters from command line
    part = input_partition()

    # create input dictionary
    d = process_inputs(part)

    # save various output files
    save_files(d, part, VALUES_MODEL)


if __name__ == '__main__':
    main()
