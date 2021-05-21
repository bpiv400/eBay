from shutil import rmtree
import pandas as pd
from rlenv.generate.util import process_sims
from utils import unpickle, topickle, input_partition
from constants import SIM_DIR, NUM_CHUNKS
from featnames import LSTG


def process_values_sims(sims=None):
    s = pd.concat(sims).sort_index()
    sale = s.apply(lambda x: x > 0)
    num_sales = sale.groupby(LSTG).sum()
    num_exps = (~sale).groupby(LSTG).sum()
    psale = num_sales / (num_sales + num_exps)
    price = s[sale].groupby(LSTG).mean()
    price = price.reindex(index=psale.index, fill_value=0)
    df = pd.concat([price.rename('x'), psale.rename('p')],
                   axis=1).sort_index()
    return df


def main():
    part, values = input_partition(agent=True, opt_arg='values')

    output_dir = SIM_DIR + '{}/'.format(part)
    sub_dir = output_dir + '{}/'.format('values' if values else 'outcomes')

    # concatenate
    sims = []
    for i in range(NUM_CHUNKS):
        chunk_path = sub_dir + '{}.pkl'.format(i)
        sims.append(unpickle(chunk_path))

    # concatenate, clean, and save
    if values:
        df = process_values_sims(sims)
        topickle(df, output_dir + 'values.pkl')
    else:
        process_sims(part=part, sims=sims, sim_dir=output_dir)

    # delete individual files
    rmtree(sub_dir)


if __name__ == '__main__':
    main()
