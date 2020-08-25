import argparse
import torch
import pandas as pd
from agent.values.ValueCalculator import ValueCalculator
from agent.values.ValueRecorder import ValueRecorder
from agent.eval.generate import AgentGenerator
from agent.util import get_paths, load_agent_model
from utils import run_func_on_chunks, process_chunk_worker, topickle
from constants import RL_SLR, TEST
from featnames import START_PRICE


class ValueGenerator(AgentGenerator):

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
    parser.add_argument('--name', type=str)
    args = parser.parse_args()

    # environment class and run directory
    _, _, run_dir = get_paths(byr=False, name=args.name)

    # recreate model
    nocon = 'nocon' in args.name
    model = load_agent_model(
        model_args=dict(byr=False, nocon=nocon),
        run_dir=run_dir
    )

    # run in parallel on chunks
    sims = run_func_on_chunks(
        f=process_chunk_worker,
        func_kwargs=dict(
            part=args.part,
            gen_class=ValueGenerator,
            gen_kwargs=dict(model=model)
        )
    )

    # combine and process output
    output_dir = run_dir + '{}/'.format(args.part)
    df = pd.concat(sims, axis=0).sort_index()
    topickle(df, output_dir + 'values.pkl')


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
