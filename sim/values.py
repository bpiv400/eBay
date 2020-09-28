from datetime import datetime as dt
import os
import torch
import pandas as pd
from sim.envs import SimulatorEnv
from sim.generate import OutcomeGenerator
from utils import run_func_on_chunks, process_chunk_worker, input_partition, topickle
from constants import SIM_DIR, VALUE_SIMS
from featnames import LSTG

COLS = [LSTG, 'sale', 'sale_price', 'relist_ct']


class ValueGenerator(OutcomeGenerator):

    def generate_recorder(self):
        return None

    def generate(self):
        rows = []
        while self.env.has_next_lstg():
            start = dt.now()
            lstg = self.env.next_lstg()
            for i in range(VALUE_SIMS):
                row = self.simulate_lstg()
                rows.append([lstg, i] + list(row))

            # print listing summary
            elapsed = int(round((dt.now() - start).total_seconds()))
            print('{}: {} sec'.format(lstg, elapsed))

        # convert to dataframe
        df = pd.DataFrame.from_records(rows, columns=COLS)
        df = df.set_index([LSTG, 'sale']).sort_index()
        return df

    def simulate_lstg(self):
        """
        Simulate until sale.
        :return: (sale price, relist count)
        """
        relist_ct = -1
        while relist_ct < 1000:
            relist_ct += 1
            self.env.reset()
            self.env.run()
            if self.env.outcome.sale:
                break
        return self.env.outcome.price, relist_ct


def main():
    part = input_partition()

    # process chunks in parallel
    sims = run_func_on_chunks(
        f=process_chunk_worker,
        func_kwargs=dict(part=part,
                         gen_class=ValueGenerator,
                         gen_kwargs=dict(env=SimulatorEnv))
    )

    # concat and save output
    df = pd.concat(sims, axis=0).sort_index()
    output_dir = SIM_DIR + '{}/'.format(part)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    topickle(df, output_dir + 'values.pkl')


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('forkserver')
    main()
