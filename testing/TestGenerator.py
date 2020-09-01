from sim.generate import SimulatorEnv
from rlenv.Composer import Composer
from rlenv.generate.Generator import Generator
from rlenv.util import get_env_sim_subdir
from testing.Listing import Listing
from testing.TestQueryStrategy import TestQueryStrategy
from testing.util import load_all_inputs, reindex_dict, \
    get_auto_safe_lstgs, drop_duplicated_timestamps
from testing.TestLoader import TestLoader
from utils import unpickle, load_file
from agent.const import FULL
from featnames import X_THREAD, X_OFFER, LOOKUP, LSTG


class TestGenerator(Generator):
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose, byr=False, slr=False)
        print('Testing Simulator')

    def generate_env(self):
        return self.env_class(query_strategy=self.query_strategy,
                              loader=self.loader,
                              recorder=None,
                              verbose=self.verbose,
                              composer=self.composer,
                              con_set=FULL,
                              test=True)

    def generate_query_strategy(self):
        return TestQueryStrategy()

    def generate_composer(self):
        return Composer(cols=self.loader.x_lstg_cols)

    def load_chunk(self, chunk=None, part=None):
        """
        Initializes lstg loader for the
        """
        print('Loading testing data...')
        chunk_dir = get_env_sim_subdir(part=part, chunks=True)
        chunk_path = chunk_dir + '{}.pkl'.format(chunk)
        chunk = unpickle(chunk_path)
        lstgs = chunk[LOOKUP].index

        # add in threads and offers
        for name in [X_THREAD, X_OFFER]:
            chunk[name] = load_file(part, name).reindex(
                index=lstgs, level=LSTG)

        # model inputs, subset to lstgs
        inputs = load_all_inputs(part=part, byr=self.byr, slr=self.slr)
        inputs = reindex_dict(d=inputs, lstgs=lstgs)

        # subset to valid listings
        valid = self._get_valid_lstgs(part=part, chunk=chunk)
        if len(valid) < len(lstgs):
            chunk = reindex_dict(d=chunk, lstgs=valid)
            inputs = reindex_dict(d=inputs, lstgs=valid)
        print('{} listings in chunk, {} of them valid'.format(
            len(lstgs), len(valid)))

        # put in dictionary
        chunk['inputs'] = inputs

        print('Running tests...')
        return TestLoader(chunk)

    def _get_valid_lstgs(self, part=None, chunk=None):
        """
        Retrieves a list of lstgs from the chunk where the agents makes at least one
        turn.

        Verifies this list matches exactly the lstgs with inputs for the relevant model
        :return: pd.Int64Index
        """
        # lstgs without duplicated time stamps first
        non_dups = drop_duplicated_timestamps(part=part, chunk=chunk)
        auto_safe = get_auto_safe_lstgs(chunk)
        return non_dups.intersection(auto_safe, sort=None)

    def _get_listing_params(self):
        params = {
            LSTG: self.loader.lstg,
            'inputs': self.loader.inputs,
            X_THREAD: self.loader.x_thread,
            X_OFFER: self.loader.x_offer,
            LOOKUP: self.loader.lookup,
            'verbose': self.verbose
        }
        return params

    def generate(self):
        while self.env.has_next_lstg():
            self.env.next_lstg()
            self.simulate_lstg()

    def simulate_lstg(self):
        """
        Simulates a particular listing once
        :return: outcome tuple
        """
        params = self._get_listing_params()
        lstg_log = Listing(params=params)
        self.query_strategy.update_log(lstg_log)
        self.env.reset()
        self.env.run()

    @property
    def env_class(self):
        return SimulatorEnv
