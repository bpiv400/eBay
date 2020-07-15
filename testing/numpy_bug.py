import numpy as np
from constants import VALIDATION
from rlenv.util import load_chunk, get_env_sim_dir

LSTG = 51043167
CHUNK = 11


def main():
    base_dir = get_env_sim_dir(VALIDATION)
    _, _, p_arrival = load_chunk(base_dir=base_dir,
                                 num=CHUNK)
    probs = p_arrival.loc[LSTG, :].values
    print(probs.dtype)
    print('pre norm sum: {}'.format(probs.sum()))
    probs /= probs.sum()
    print('post norm sum: {}'.format(probs))
    print(type(probs.dtype))
    np.random.choice(len(probs), p=probs)
    print('Chosen without error')


if __name__ == '__main__':
    main()
