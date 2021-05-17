import numpy as np
import pandas as pd
from rlenv.Composer import Composer
from sim.arrivals import ArrivalSimulator
from sim.chunks import ArrivalInterface, ArrivalQueryStrategy
from utils import topickle, load_data, load_file, safe_reindex
from constants import PARTS_DIR
from featnames import LSTG, THREAD, TEST, X_OFFER, X_THREAD, CLOCK, INDEX, LOOKUP, \
    START_TIME, X_LSTG, CON


def main():
    # number of buyers in the data
    data = load_data(clock=True)
    accept = (data[X_OFFER][CON] == 1).groupby(LSTG).max()
    sales = accept[accept].index
    num_buyers = data[X_THREAD].iloc[:, 0].groupby(LSTG).count().rename('arrivals')

    # initialize arrivals simulator
    composer = Composer()
    qs = ArrivalQueryStrategy(arrival=ArrivalInterface())
    simulator = ArrivalSimulator(composer=composer, query_strategy=qs)
    x_lstg = load_file(TEST, X_LSTG)

    # simulate unseen arrival process for listings that do not sell
    arrivals = data[CLOCK].xs(1, level=INDEX)
    for lstg in sales:
        if lstg not in num_buyers.index:
            continue
        i = np.argwhere(data[LOOKUP].index == lstg)[0][0]
        start_time = data[LOOKUP].loc[lstg, START_TIME]
        x_i = {k: v[i] for k, v in x_lstg.items()}
        simulator.set_lstg(start_time=start_time, x_lstg=x_i)
        arrivals_i = arrivals.xs(lstg).to_list()
        arrivals_i = simulator.simulate_arrivals(arrivals=arrivals_i)
        assert len(arrivals_i) >= num_buyers.loc[lstg]
        num_buyers.loc[lstg] = len(arrivals_i)

    # pick one buyer at random
    agent_thread = pd.Series(np.random.randint(low=1, high=num_buyers.values + 1),
                             index=num_buyers.index, name=THREAD)
    agent_thread = agent_thread.to_frame().assign(temp=1).set_index(
        THREAD, append=True).index.sort_values()

    # create synthetic data
    del data[CLOCK]
    data = safe_reindex(data, idx=agent_thread, dropna=True)
    data[LOOKUP] = data[LOOKUP].loc[data[X_THREAD].index].droplevel(THREAD)

    # save
    topickle(data, PARTS_DIR + '{}/synthetic.pkl'.format(TEST))


if __name__ == '__main__':
    main()
