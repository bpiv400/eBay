import torch
import numpy as np
from env.util import model_str
from utils import load_sizes, load_featnames, load_pctile
from env.const import OFFER_MAPS, THREAD_COUNT_IND, TIME_START_IND, TIME_END_IND
from featnames import OUTCOME_FEATS, CLOCK_FEATS, TIME_FEATS, DAYS_SINCE_LSTG, \
    BYR_HIST, DELAY, CON, INT_REMAINING, DAYS_SINCE_LAST, THREAD_COUNT, SLR, \
    OFFER_MODELS, MODELS, FIRST_ARRIVAL_MODEL, INTERARRIVAL_MODEL, BYR_HIST_MODEL, \
    X_LSTG, LSTG, CLOCK


class Composer:
    """
    Class for composing inputs to interface from various input streams
    """
    def __init__(self):
        self.sizes = {m: load_sizes(m) for m in MODELS}
        self.x_lstg_cols = load_featnames(X_LSTG)
        self.lstg_sets = self.build_lstg_sets()

        # for hist model
        s = load_pctile(name=BYR_HIST)
        s = s.reindex(index=range(s.index.max() + 1), method='pad')
        self.hist_pctile = s

    def hist_to_pctile(self, hist=None):
        """
        Converts integer number of past listings into a percentile.
        :param int hist: number of prior Best Offer listings.
        :return: float percentile
        """
        if hist > self.hist_pctile.index[-1]:
            return 1.
        return self.hist_pctile.loc[hist]

    def build_lstg_sets(self):
        """
        Constructs a dictionary containing the input groups constructed
        from the features in x_lstg
        :return: dict
        """
        featnames = load_featnames(FIRST_ARRIVAL_MODEL)
        for c in featnames[LSTG]:
            assert c in self.x_lstg_cols[LSTG]
        featnames[SLR] = load_featnames(model_str(CON, turn=2))[SLR]
        for model in MODELS:
            # verify all x_lstg based sets contain the same features in the same order
            self.verify_lstg_sets_shared(model, featnames.copy())
            if model in OFFER_MODELS:
                Composer.verify_offer_feats(model)
                if DELAY not in model:
                    Composer.verify_offer_append(model, featnames[LSTG])
                else:
                    Composer.verify_delay_append(model, featnames[LSTG])
            elif model == FIRST_ARRIVAL_MODEL:
                Composer.verify_first_arrival_append(featnames[LSTG])
            elif model == INTERARRIVAL_MODEL:
                Composer.verify_interarrival_append(featnames[LSTG])
            else:
                Composer.verify_hist_append(featnames[LSTG])
        return featnames

    @staticmethod
    def verify_offer_feats(model):
        turn = int(model[-1])
        if turn % 2 == 0:
            assumed_feats = CLOCK_FEATS + TIME_FEATS + OUTCOME_FEATS
        else:
            assumed_feats = CLOCK_FEATS + OUTCOME_FEATS
        model_feats = load_featnames(model)['offer']
        Composer.verify_all_feats(assumed_feats=assumed_feats,
                                  model_feats=model_feats)
        model_sizes = load_sizes(model)
        for j in range(1, 8):
            if j < turn or (j == turn and DELAY not in model):
                assert 'offer{}'.format(j) in model_sizes['x']
            else:
                assert 'offer{}'.format(j) not in model_sizes['x']

    def verify_lstg_sets_shared(self, model, featnames):
        """
        Ensures that all the input groupings that contain features from x_lstg
        have a common ordering
        :param str model: model name
        :param dict featnames: dictionary containing all x_lstg features
        that appear in the model organized into feature groupings
        :return: None
        """
        model_featnames = load_featnames(model)
        missing_idx = list()
        # check that all features in LSTG not in x_lstg are appended to the end of LSTG
        for feat in model_featnames[LSTG]:
            if feat not in self.x_lstg_cols[LSTG]:
                missing_idx.append(model_featnames[LSTG].index(feat))
        if len(missing_idx) != 0:
            missing_idx_min = min(missing_idx)
            assert missing_idx_min == len(featnames[LSTG])
        # remove those missing features
        model_featnames[LSTG] = [feat for feat in model_featnames[LSTG]
                                 if feat in self.x_lstg_cols[LSTG]]
        # iterate over all x_lstg features based and check that have same elements in the same order
        for grouping_name, lstg_feats in featnames.items():
            model_grouping = model_featnames[grouping_name]
            assert len(model_grouping) == len(lstg_feats)
            for model_feat, lstg_feat in zip(model_grouping, lstg_feats):
                assert model_feat == lstg_feat
                assert model_feat in self.x_lstg_cols[grouping_name]

    def build_input_dict(self, model_name, sources=None, turn=None):
        """
        Public method that composes input vectors (x_time and x_fixed) from tensors in the
        environment

        :param model_name: str giving the name of the focal model
        :param sources: dictionary containing tensors from the environment that contain
        features the model expects in the input
        :param turn: turn number
        :return: dict
        """
        input_dict = dict()
        fixed_sizes = self.sizes[model_name]['x']  # dict
        for input_set, size in fixed_sizes.items():
            if input_set == LSTG:
                input_dict[input_set] = self._build_lstg_vector(
                    model_name, sources=sources)
            elif 'offer' == input_set[:-1]:
                input_dict[input_set] = self._build_offer_vector(
                    offer_vector=sources[input_set], byr=turn % 2 != 0)
            else:
                input_dict[input_set] = torch.from_numpy(sources[input_set]).float().unsqueeze(0)
        return input_dict

    @staticmethod
    def _build_offer_vector(offer_vector, byr=False):
        if not byr:
            full_vector = offer_vector
        else:
            full_vector = np.concatenate([offer_vector[:TIME_START_IND],
                                          offer_vector[TIME_END_IND:]])
        return torch.from_numpy(full_vector).unsqueeze(0).float()

    @staticmethod
    def _build_lstg_vector(model_name, sources=None):
        if model_name == FIRST_ARRIVAL_MODEL:
            lstg = sources[LSTG]  # append nothing
        elif model_name == INTERARRIVAL_MODEL:
            solo_feats = np.array([sources[DAYS_SINCE_LSTG],
                                   sources[DAYS_SINCE_LAST],
                                   sources[THREAD_COUNT]])
            lstg = np.concatenate([sources[LSTG],
                                   sources[CLOCK],
                                   solo_feats])
        elif model_name == BYR_HIST_MODEL:
            solo_feats = np.array([sources[DAYS_SINCE_LSTG],
                                   sources[THREAD_COUNT] - 1])
            lstg = np.concatenate([sources[LSTG],
                                   sources[CLOCK],
                                   solo_feats])
        else:
            solo_feats = [sources[DAYS_SINCE_LSTG],
                          sources[BYR_HIST],
                          sources[OFFER_MAPS[1]][THREAD_COUNT_IND] + 1]
            if DELAY in model_name:
                solo_feats += [sources[INT_REMAINING]]
            lstg = np.concatenate([sources[LSTG], solo_feats])
        lstg = lstg.astype(np.float32)
        return torch.from_numpy(lstg).float().unsqueeze(0)

    @staticmethod
    def remove_shared_feats(model_feats, shared_feats):
        return model_feats[len(shared_feats):]

    @staticmethod
    def verify_sequence(model_feats, seq_feats, start_idx):
        subset = model_feats[start_idx:(len(seq_feats) + start_idx)]
        for model_feat, seq_feat in zip(subset, seq_feats):
            assert model_feat == seq_feat

    @staticmethod
    def verify_offer_append(model, shared_feats):
        model_feats = load_featnames(model)[LSTG]
        model_feats = Composer.remove_shared_feats(model_feats, shared_feats)
        assert model_feats[0] == DAYS_SINCE_LSTG
        assert model_feats[1] == BYR_HIST
        assert model_feats[2] == THREAD_COUNT
        assert len(model_feats) == 3

    @staticmethod
    def verify_all_feats(assumed_feats=None, model_feats=None):
        assert len(model_feats) == len(assumed_feats)
        for exp_feat, model_feat in zip(assumed_feats, model_feats):
            assert exp_feat == model_feat

    @staticmethod
    def verify_delay_append(model, shared_feats):
        model_feats = load_featnames(model)[LSTG]
        model_feats = Composer.remove_shared_feats(model_feats, shared_feats)
        assert model_feats[0] == DAYS_SINCE_LSTG
        assert model_feats[1] == BYR_HIST
        assert model_feats[2] == THREAD_COUNT
        assert model_feats[3] == INT_REMAINING
        assert len(model_feats) == 4

    @staticmethod
    def verify_interarrival_append(shared_feats):
        model_feats = load_featnames(INTERARRIVAL_MODEL)[LSTG]
        model_feats = Composer.remove_shared_feats(model_feats, shared_feats)
        Composer.verify_sequence(model_feats, CLOCK_FEATS, 0)
        next_ind = len(CLOCK_FEATS)
        assert model_feats[next_ind] == DAYS_SINCE_LSTG
        assert model_feats[next_ind + 1] == DAYS_SINCE_LAST
        assert model_feats[next_ind + 2] == THREAD_COUNT

    @staticmethod
    def verify_first_arrival_append(shared_feats):
        model_feats = load_featnames(FIRST_ARRIVAL_MODEL)[LSTG]
        model_feats = Composer.remove_shared_feats(model_feats, shared_feats)
        assert len(model_feats) == 0

    @staticmethod
    def verify_hist_append(shared_feats):
        model_feats = load_featnames(BYR_HIST_MODEL)[LSTG]
        model_feats = Composer.remove_shared_feats(model_feats, shared_feats)
        Composer.verify_sequence(model_feats, CLOCK_FEATS, 0)
        assert model_feats[len(CLOCK_FEATS)] == DAYS_SINCE_LSTG
        assert model_feats[len(CLOCK_FEATS) + 1] == THREAD_COUNT
