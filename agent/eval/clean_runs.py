import os
import shutil
import argparse
from compress_pickle import load
from constants import AGENTS, REINFORCE_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', choices=AGENTS, required=True)
    name = parser.parse_args().name

    parent_dir = REINFORCE_DIR + '{}/'.format(name)
    runs = load(parent_dir + 'runs.pkl')
    folders = [f for f in os.listdir(parent_dir) if os.path.isdir(parent_dir + f)]
    for folder in folders:
        run_id = folder[4:]
        if run_id not in runs.index:
            shutil.rmtree(parent_dir + '{}/'.format(folder))


if __name__ == '__main__':
    main()
