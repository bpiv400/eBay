import numpy as np
from functools import reduce
from agent.const import BYR, FEAT_TYPE, ALL_FEATS
from agent.util import get_agent_name
from agent.AgentComposer import AgentComposer
from constants import POLICY_MODELS
from featnames import LSTG, BYR_HIST, ACC_PRICE, DEC_PRICE, START_PRICE
from rlenv.Composer import Composer
from rlenv.environments.BuyerEnvironment import BuyerEnvironment
from rlenv.environments.SellerEnvironment import SellerEnvironment
from rlenv.environments.SimulatorEnvironment import SimulatorEnvironment
from rlenv.generate.Generator import Generator
from rlenv.util import get_env_sim_subdir, load_chunk
from rlenv.generate.Recorder import OutcomeRecorder
from testing.LstgLog import LstgLog
from testing.TestQueryStrategy import TestQueryStrategy
from testing.util import subset_inputs, load_all_inputs, load_reindex, \
    lstgs_without_duplicated_timestamps, subset_lstgs
from testing.TestLoader import TestLoader
from utils import init_optional_arg


class TestGenerator(Generator):
    def __init__(self, **kwargs):
        """
        kwargs contains start, role, agent, delay, part, verbose
        :param kwargs:
        """
        super().__init__(verbose=kwargs['verbose'])
        init_optional_arg(kwargs=kwargs, name='start',
                          default=None)
        # id of the first lstg in the chunk to simulate
        self.start = kwargs['start']
        # boolean for whether the testing is for an agent environment
        self.agent = kwargs['agent']
        # boolean for whether the agent is a byr
        self.byr = kwargs[BYR]

    def generate_query_strategy(self):
        return TestQueryStrategy()

    def generate_composer(self):
        if self.agent:
            agent_params = {
                BYR: self.byr,
                FEAT_TYPE: ALL_FEATS,
                BYR_HIST: None
            }
            return AgentComposer(agent_params=agent_params)
        else:
            return Composer(cols=self.loader.x_lstg_cols)

    def generate_recorder(self):
        return OutcomeRecorder(verbose=self.verbose)

    def load_chunk(self, chunk=None, part=None):
        """
        Initializes lstg loader for the
        """
        print('Loading testing data...')
        chunk_dir = get_env_sim_subdir(part=part, chunks=True)
        chunk_tuple = load_chunk(input_path='{}{}.gz'.format(chunk_dir, chunk))
        x_lstg, lookup, p_arrival = chunk_tuple
        # lstgs without duplicated time stamps first
        test_data = dict()
        non_dups = lstgs_without_duplicated_timestamps(
            lstgs=lookup.index)
        x_lstg = x_lstg.reindex(non_dups)
        lookup = lookup.reindex(non_dups)
        p_arrival = p_arrival.reindex(non_dups)
        assert 59729927 not in lookup
        test_data['inputs'] = load_all_inputs(part=part,
                                              lstgs=lookup.index)
        for name in ['x_thread', 'x_offer']:
            test_data[name] = load_reindex(part=part,
                                           name=name,
                                           lstgs=lookup.index)
        test_data = self._remove_extra_models(test_data=test_data)
        # subset inputs to only contain lstgs where the agent has at least 1 action
        valid_lstgs = self._get_valid_lstgs(test_data=test_data, lookup=lookup)
        if len(valid_lstgs) < len(lookup.index):
            x_lstg = subset_lstgs(df=x_lstg, lstgs=valid_lstgs)
            lookup = subset_lstgs(df=lookup, lstgs=valid_lstgs)
            p_arrival = subset_lstgs(df=p_arrival, lstgs=valid_lstgs)
            test_data['x_thread'] = subset_lstgs(df=test_data['x_thread'], lstgs=valid_lstgs)
            test_data['x_offer'] = subset_lstgs(df=test_data['x_offer'], lstgs=valid_lstgs)
            test_data['inputs'] = subset_inputs(input_data=test_data['inputs'],
                                                value=valid_lstgs, level='lstg')
        return TestLoader(x_lstg=x_lstg, lookup=lookup,
                          p_arrival=p_arrival, test_data=test_data,
                          agent=self.agent)

    def _remove_extra_models(self, test_data):
        """
        Remove the unused policy models from the testing data inputs
        """
        unused_models = POLICY_MODELS
        if self.agent:
            agent_name = get_agent_name(byr=self.byr)
            unused_models.remove(agent_name)
        for model_name in unused_models:
            del test_data['inputs'][model_name]
        return test_data

    @staticmethod
    def _get_auto_safe_lstgs(test_data=None, lookup=None):
        """
        Drops lstgs where there's at least one offer within 1%
        of the accept or reject price if that price is non-null
        """
        print('Lstg count: {}'.format(len(lookup)))
        lookup = lookup.copy()
        # normalize start / decline prices
        for price in [ACC_PRICE, DEC_PRICE]:
            lookup[price] = lookup[price] / lookup[START_PRICE]
        # drop offers that are 0 or 1 (and very near or 1)
        offers = test_data['x_offer'].copy()
        near_null_offers = (offers['norm'] >= .99) | (offers['norm'] <= 0.01)
        offers = offers.loc[~near_null_offers, :]

        # extract lstgs that no longer appear in offers
        # these should be kept because they  must have no offers
        # or consist entirely of offers adjacent to null acc/rej prices
        null_offer_lstgs = lookup.index[~lookup.index.isin(
            offers.index.get_level_values('lstg').unique())]

        # inner join offers with lookup
        offers = offers.join(other=lookup, on='lstg')
        offers['diff_acc'] = (offers['norm'] - offers[ACC_PRICE]).abs()
        offers['diff_dec'] = (offers['norm'] - offers[DEC_PRICE]).abs()
        offers['low_diff'] = (offers['diff_acc'] < 0.01) |\
                             (offers['diff_dec'] < 0.01)
        low_diff_count = offers['low_diff'].groupby(level='lstg').sum()
        no_low_diffs_lstgs = low_diff_count.index[low_diff_count == 0]
        output_lstgs = no_low_diffs_lstgs.union(null_offer_lstgs)
        print('num lstgs after diffs removal: {}'.format(
            len(output_lstgs)))
        return output_lstgs

    @staticmethod
    def _get_delay_buyer_lstgs(test_data=None, lookup=None):
        """
        Drops lstgs where a buyer makes at least one instantaneous offer
        """
        print('Lstg count: {}'.format(len(lookup.index)))
        lookup = lookup.copy()
        offers = test_data['x_offer'].copy()
        offers['byr'] = offers.index.get_level_values('index') % 2 == 1
        offers['not_first'] = (offers.index.get_level_values('index') != 1)
        offers['byr_not_first'] = (offers['byr']) & (offers['not_first'])
        offers['instant_byr'] = (offers['byr_not_first']) & (offers['delay'] == 0)
        instant_count = offers['instant_byr'].groupby(level='lstg').sum()
        instant_lstgs = instant_count.index[instant_count > 0]
        lookup = lookup.drop(index=instant_lstgs, inplace=False)
        print('num lstgs after instant removal: {}'.format(
            len(lookup.index)
        ))
        return lookup.index

    def _get_valid_lstgs(self, test_data=None, lookup=None):
        """
        Retrieves a list of lstgs from the chunk where the agent makes at least one
        turn.

        Verifies this list matches exactly the lstgs with inputs for the relevant model
        :return: pd.Int64Index
        """
        auto_safe_lstgs = self._get_auto_safe_lstgs(test_data=test_data,
                                                    lookup=lookup)
        non_instant_lstgs = self._get_delay_buyer_lstgs(test_data=test_data,
                                                        lookup=lookup)
        if self.start is not None:
            start_index = list(lookup.index).index(self.start)
            start_lstgs = lookup.index[start_index:]
        else:
            start_lstgs = lookup.index
        lstg_groups = [
            start_lstgs,
            auto_safe_lstgs,
            non_instant_lstgs
        ]
        if self.agent:
            agent_lstgs = self._get_agent_lstgs(test_data=test_data)
            lstg_groups.append(agent_lstgs)
        return reduce(np.intersect1d, lstg_groups)

    def _get_agent_lstgs(self, test_data=None):
        x_offer = test_data['x_offer'].copy()  # x_offer: pd.DataFrame
        if self.byr:
            # all lstgs should have at least 1 action
            lstgs = x_offer.index.get_level_values('lstg').unique()
        else:
            # keep all lstgs with at least 1 thread with at least 1 non-auto seller
            # offer
            slr_offers = x_offer.index.get_level_values('index') % 2 == 0
            man_offers = ~x_offer['auto']
            predicates = (slr_offers, man_offers)
            keep = np.logical_and.reduce(predicates)
            lstgs = x_offer.index.get_level_values('lstg')[keep].unique()
        # verify that x_offer based lstgs match lstgs used as model input exactly
        model_name = get_agent_name(byr=self.byr)
        input_lstgs = test_data['inputs'][model_name][LSTG].index.get_level_values('lstg').unique()
        # print(input_lstgs[~input_lstgs.isin(lstgs)])
        assert input_lstgs.isin(lstgs).all()
        # print(lstgs[~lstgs.isin(input_lstgs)])
        # assert lstgs.isin(input_lstgs).all()
        return lstgs

    def _count_rl_buyers(self):
        # there should always be at least 1 if initial agent subsetting
        # was correct
        threads = self.loader.x_offer.index.get_level_values('thread')
        return len(threads.unique())

    def _generate_lstg_log(self, buyer=None):
        params = {
            'lstg': self.loader.lstg,
            'inputs': self.loader.inputs,
            'x_thread': self.loader.x_thread,
            'x_offer': self.loader.x_offer,
            'lookup': self.loader.lookup
        }
        if self.agent:
            params['agent_params'] = {
                'byr': self.byr,
                'thread_id': buyer
            }
        return LstgLog(params=params)

    def simulate_agent_lstg(self, buyer=None):
        lstg_log = self._generate_lstg_log(buyer=buyer)
        self.query_strategy.update_log(lstg_log)
        if self.byr:
            hist = self.loader.x_thread.loc[buyer, 'byr_hist']
            hist = hist / 10
            self.composer.set_hist(hist=hist)
        # print('resetting: {}'.format(self.loader.lstg))
        obs = self.environment.reset()
        agent_tuple = obs, None, None, None
        done = False
        while not done:
            action = lstg_log.get_action(agent_tuple=agent_tuple)
            agent_tuple = self.environment.step(action)
            done = agent_tuple[2]
        lstg_log.verify_done()

    def generate(self):
        while self.environment.has_next_lstg():
            print('next lstg')
            self.environment.next_lstg()
            self.recorder.update_lstg(lookup=self.loader.lookup,
                                      lstg=self.loader.lstg)
            if self.byr:
                buyers = self._count_rl_buyers()
                for i in range(buyers):
                    # simulate lstg once for each buyer
                    self.simulate_agent_lstg(buyer=(i + 1))
            elif self.agent:
                self.simulate_agent_lstg()
            else:
                self.simulate_lstg()

    def simulate_lstg(self):
        """
        Simulates a particular listing once
        :return: outcome tuple
        """
        # simulate once for each buyer in BuyerEnvironment mode
        # this output needs to match first agent action
        lstg_log = self._generate_lstg_log(buyer=None)
        self.query_strategy.update_log(lstg_log)
        self.environment.reset()
        # going to need to change to return for each agent action
        outcome = self.environment.run()
        return outcome

    @property
    def env_class(self):
        if self.agent:
            if self.byr:
                env_class = BuyerEnvironment
            else:
                env_class = SellerEnvironment
        else:
            env_class = SimulatorEnvironment
        return env_class

    @property
    def records_path(self):
        raise RuntimeError("No recorder")
