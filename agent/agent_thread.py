import numpy as np
import pandas as pd
from utils import unpickle, topickle, input_partition
from constants import NUM_CHUNKS, PARTS_DIR, OUTCOME_SIMS
from featnames import ARRIVALS, LSTG, SIM, THREAD


def sample_agent(part=None):
    # number of buyers per simulation
    toconcat = []
    for i in range(NUM_CHUNKS):
        path = PARTS_DIR + '{}/chunks/{}.pkl'.format(part, i)
        arrivals = unpickle(path)[ARRIVALS]
        for lstg, sims in arrivals.items():
            ct = [len(e) for e in sims[:OUTCOME_SIMS]]
            idx = pd.MultiIndex.from_product([[lstg], range(OUTCOME_SIMS)],
                                             names=[LSTG, SIM])
            s = pd.Series(ct, name=THREAD, index=idx)
            s = s[s > 0]  # throw out simulations with zero arrivals
            toconcat.append(s)
    s = pd.concat(toconcat, axis=0)

    # pick one buyer at random
    agent_thread = pd.Series(np.random.randint(low=1, high=s.values+1),
                             index=s.index, name=THREAD)
    agent_thread = agent_thread.to_frame().assign(temp=1).set_index(
        THREAD, append=True).index.sort_values()

    return agent_thread


def main():
    part = input_partition(agent=True)
    agent_thread = sample_agent(part)
    topickle(agent_thread, PARTS_DIR + '{}/agent_thread.pkl'.format(part))


if __name__ == '__main__':
    main()
