import argparse
import torch
import pandas as pd
from agent.values.ValueCalculator import ValueCalculator
from agent.values.ValueRecorder import ValueRecorder
from agent.eval.generate import AgentGenerator, load_agent_model
from agent.util import get_paths
from utils import run_func_on_chunks, process_chunk_worker, topickle
from agent.const import FULL, SPARSE, NOCON
from constants import RL_SLR, TEST
from featnames import START_PRICE


class ValueGenerator(AgentGenerator):
    def __init__(self, model=None, con_set=None):
        super().__init__(verbose=False, byr=False, slr=True,
                         model=model, con_set=con_set)

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
    parser.add_argument('--part', type=str, choices=[RL_SLR, TEST])
    parser.add_argument('--con_set', choices=[FULL, SPARSE, NOCON])
    parser.add_argument('--name', type=str)
    args = parser.parse_args()

    # environment class and run directory
    _, _, run_dir = get_paths(byr=False, name=args.name)

    # recreate model
    model = load_agent_model(
        model_args=dict(byr=False, con_set=args.con_set),
        run_dir=run_dir
    )

    # run in parallel on chunks
    sims = run_func_on_chunks(
        f=process_chunk_worker,
        func_kwargs=dict(
            part=args.part,
            gen_class=ValueGenerator,
            gen_kwargs=dict(model=model, con_set=args.con_set)
        )
    )

    # combine and process output
    output_dir = run_dir + '{}/'.format(args.part)
    df = pd.concat(sims, axis=0).sort_index()
    topickle(df, output_dir + 'values.pkl')


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
