import numpy as np
import pandas as pd
from agent.util import get_sim_dir, load_valid_data
from utils import safe_reindex
from agent.const import BYR_CONS
from featnames import X_OFFER, CON, INDEX, AUTO, EXP, LSTG, SIM, THREAD, NORM


def collect_sims(mask=None):
    indices = mask[mask].index
    toAppend = []
    for i, index in enumerate(indices):
        sim_dir = get_sim_dir(byr=True, heuristic=True, index=index)
        data = load_valid_data(sim_dir=sim_dir, minimal=True, lookup=False)
        df = data[X_OFFER][[CON, NORM]].reset_index().drop(SIM, axis=1)
        df[SIM] = i
        df.set_index([LSTG, SIM, THREAD, INDEX], inplace=True)
        toAppend.append(df)
    df = pd.concat(toAppend).sort_index()
    return df


def main():
    # .5 -> 0 -> .2 vs. .6
    mask = (BYR_CONS[1] == .5) & (BYR_CONS[3] == .2)
    df = collect_sims(mask)

    print('Accept rate: {}'.format((df[CON].xs(2, level=INDEX) == 1).mean()))

    df2 = df.xs(2, level=INDEX)
    rej2 = df2.loc[~df2[AUTO] & ~df2[EXP], CON] == 0
    idx = rej2[rej2].index
    acc4 = df[CON].xs(4, level=INDEX) == 1
    idx = idx.intersection(acc4.index)
    acc4 = acc4.loc[idx]
    print(acc4.mean())

    mask = BYR_CONS[1] == .6
    df = collect_sims(mask)
    print('Accept rate: {}'.format((df[CON].xs(2, level=INDEX) == 1).mean()))


if __name__ == '__main__':
    main()
