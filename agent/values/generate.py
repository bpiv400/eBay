import argparse
import torch
import pandas as pd
from agent.values.ValueCalculator import ValueCalculator
from agent.values.ValueRecorder import ValueRecorder
from agent.eval.generate import AgentGenerator, load_agent_model
from assess.util import find_best_run
from utils import run_func_on_chunks, process_chunk_worker, topickle
from constants import RL_BYR, TEST
from featnames import START_PRICE


class ValueGenerator(AgentGenerator):
    def __init__(self, model=None, con_set=None):
        super().__init__(byr=False, slr=True, model=model, con_set=con_set)

    def generate_recorder(self):
        return ValueRecorder(verbose=self.verbose)

    def simulate_lstg(self):
        # initialize value calculator
        val_calc = ValueCalculator(cut=self.env.cut,
                                   start_price=self.env.lookup[START_PRICE])

        # simulate lstg until a stopping criterion is satisfied
        while not val_calc.stabilized:
            super().simulate_lstg()
            val_calc.add_outcome(self.env.outcome.price)

        # save results to value calculator
        self.recorder.add_val(val_calc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str, choices=[RL_BYR, TEST])
    part = parser.parse_args().part

    # directory of best seller run
    run_dir = find_best_run(byr=False)
    _, con_set, dropout0, dropout1 = run_dir.split('/')[-2].split('_')
    dropout = (float(dropout0) / 10, float(dropout1) / 10)

    # recreate model
    model = load_agent_model(
        model_args=dict(byr=False, dropout=dropout, con_set=con_set),
        run_dir=run_dir
    )

    # run in parallel on chunks
    sims = run_func_on_chunks(
        f=process_chunk_worker,
        func_kwargs=dict(
            part=part,
            gen_class=ValueGenerator,
            gen_kwargs=dict(model=model, con_set=con_set)
        )
    )

    # combine and process output
    output_dir = run_dir + '{}/'.format(part)
    df = pd.concat(sims, axis=0).sort_index()
    topickle(df, output_dir + 'values.pkl')


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
