import pytest
from rlenv.composer.maps import *
from rlenv.composer.Composer import Composer
from rlenv.env_consts import *
from rlenv.env_utils import *
from rlenv.interface import model_names


@pytest.fixture
def new_composer():
    """
    Builds a composer from scratch

    :return: rlenv.Composer
    """
    composer = Composer({'composer': 1}, rebuild=True)
    return composer


@pytest.fixture
def composer():
    """
    Loads a composer from the set of stored dictionaries

    :return: rlenv.Composer
    """
    composer = Composer({'composer': 1}, rebuild=False)
    return composer


def get_offer_fixed(composer):
    sources = dict()
    start = 0
    end = len(composer.x_lstg_cols)
    sources[LSTG_MAP] = torch.arange(start, end).float()
    start = end
    end += 1
    sources[BYR_HIST_MAP] = torch.arange(start, end).float()
    start = end
    end += 1
    sources[MONTHS_LSTG_MAP] = torch.tensor([start]).float()
    return sources


def byr_early_offer_sources():
    sources = dict()
    start = 0
    tot = 2
    sources[OUTCOMES_MAP] = torch.arange(start, len(SLR_OUTCOMES)).float()
    start = tot
    tot += len(CLOCK_FEATS)
    sources[CLOCK_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[DIFFS_MAP] = torch.arange(start, tot).float()
    sources[O_OUTCOMES_MAP] = torch.tensor([23, 24, 33, 34, 35, 36, 37, 38, 39]).float()
    start = 25
    tot = start + len(CLOCK_FEATS)
    sources[O_CLOCK_MAP] = torch.arange(start, tot).float()
    start = 40
    tot = start + len(TIME_FEATS)
    sources[O_DIFFS_MAP] = torch.arange(start, tot).float()
    start = 57
    tot = 60
    sources[TURN_IND_MAP] = torch.arange(start, tot).float()
    return sources


def slr_early_offer_sources():
    sources = dict()
    start = 0
    tot = 2
    sources[OUTCOMES_MAP] = torch.arange(start, len(BYR_OUTCOMES)).float()
    start = tot
    tot += len(CLOCK_FEATS)
    sources[CLOCK_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[DIFFS_MAP] = torch.arange(start, tot).float()
    sources[O_OUTCOMES_MAP] = torch.tensor([23, 24, 33, 34, 35, 36]).float()
    start = 25
    tot = start + len(CLOCK_FEATS)
    sources[O_CLOCK_MAP] = torch.arange(start, tot).float()
    start = 37
    tot = start + len(TIME_FEATS)
    sources[O_DIFFS_MAP] = torch.arange(start, tot).float()
    start = 57
    tot = 59
    sources[TURN_IND_MAP] = torch.arange(start, tot).float()
    return sources


def test_num_offers_fixed(composer):
    arr_feats = load_featnames(ARRIVAL_PREFIX, model_names.NUM_OFFERS)['x_fixed']
    targ = torch.arange(len(arr_feats)).float()
    x_fixed = composer.build_arrival_init(torch.arange(len(arr_feats)).float())
    assert torch.all(torch.eq(targ, x_fixed))


def test_num_offers_step(composer):
    arr_feats = load_featnames(ARRIVAL_PREFIX, model_names.NUM_OFFERS)
    fixed_feats = arr_feats['x_fixed']
    time_feats = arr_feats['x_time']
    sources = {
        LSTG_MAP: torch.arange(len(fixed_feats)).float(),
        CLOCK_MAP: torch.arange(len(CLOCK_FEATS)).float(),
        DIFFS_MAP: torch.arange(len(CLOCK_FEATS),
                                len(CLOCK_FEATS) + len(TIME_FEATS)).float(),
        DUR_MAP: torch.tensor([len(CLOCK_FEATS) + len(TIME_FEATS)]).float()
    }
    x_fixed, x_time = composer.build_input_vector(model_names.NUM_OFFERS,
                                                  sources=sources, fixed=True,
                                                  recurrent=True, size=1)
    assert torch.all(torch.eq(torch.arange(len(fixed_feats)).float(), x_fixed))
    assert torch.all(torch.eq(torch.arange(len(time_feats)).float(), x_time))


def test_hist_fixed(composer):
    model_name = model_names.BYR_HIST
    feats = load_featnames(ARRIVAL_PREFIX, model_name)['x_fixed']
    targ = torch.arange(len(feats)).float()
    sources = dict()
    start = 0
    end = len(composer.x_lstg_cols)
    sources[LSTG_MAP] = torch.arange(start, end).float()
    start = end
    end += len(CLOCK_FEATS)
    sources[CLOCK_MAP] = torch.arange(start, end).float()
    start = end
    end += 1
    sources[MONTHS_LSTG_MAP] = torch.tensor([start]).float()
    start = end
    end += len(TIME_FEATS)
    sources[TIME_MAP] = torch.arange(start, end).float()
    x_fixed, _ = composer.build_input_vector(model_names.BYR_HIST,
                                             sources=sources, fixed=True,
                                             recurrent=False, size=1)
    assert torch.all(torch.eq(targ, x_fixed))


def test_byr_con_fixed(composer):
    model_name = model_str(model_names.CON, byr=True)
    feats = load_featnames(BYR_PREFIX, model_names.CON)['x_fixed']
    targ = torch.arange(len(feats)).float()
    sources = get_offer_fixed(composer)
    x_fixed, _ = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                             recurrent=False, size=1)
    assert torch.all(torch.eq(targ, x_fixed))


def test_byr_con_time(composer):
    model_name = model_str(model_names.CON, byr=True)
    feats = load_featnames(BYR_PREFIX, model_names.CON)['x_time']
    targ = torch.arange(len(feats)).float()
    sources = byr_early_offer_sources()
    sources[L_OUTCOMES_MAP] = torch.tensor([-1, -1, 53, 54, 55, 56]).float()
    _, x_time = composer.build_input_vector(model_name, sources=sources, fixed=False,
                                            recurrent=True, size=1)
    assert torch.all(torch.eq(targ, x_time))


def test_byr_msg_fixed(composer):
    model_name = model_str(model_names.MSG, byr=True)
    feats = load_featnames(BYR_PREFIX, model_names.MSG)['x_fixed']
    targ = torch.arange(len(feats)).float()
    sources = get_offer_fixed(composer)
    x_fixed, _ = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                             recurrent=False, size=1)
    assert torch.all(torch.eq(targ, x_fixed))


def test_byr_msg_time(composer):
    model_name = model_str(model_names.MSG, byr=True)
    feats = load_featnames(BYR_PREFIX, model_names.MSG)['x_time']
    targ = torch.arange(len(feats)).float()
    sources = get_offer_fixed(composer)
    tot = 2
    sources[OUTCOMES_MAP] = torch.tensor([0, 1, 10, 11, 12, 13]).float()
    start = tot
    tot += len(CLOCK_FEATS)
    sources[CLOCK_MAP] = torch.arange(start, tot).float()
    start = 13
    tot = start + len(TIME_FEATS)
    sources[DIFFS_MAP] = torch.arange(start, tot).float()
    sources[O_OUTCOMES_MAP] = torch.tensor([26, 27, 36, 37, 38, 39, 40, 41, 42]).float()
    start = 28
    tot = start + len(CLOCK_FEATS)
    sources[O_CLOCK_MAP] = torch.arange(start, tot).float()
    start = 43
    tot = start + len(TIME_FEATS)
    sources[O_DIFFS_MAP] = torch.arange(start, tot).float()
    sources[L_OUTCOMES_MAP] = torch.tensor([-1, -1, -1, -1, -1, 56]).float()
    start = 57
    tot = 60
    sources[TURN_IND_MAP] = torch.arange(start, tot).float()
    _, x_time = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                            recurrent=True, size=1)
    # print(targ[~torch.eq(targ, x_time)[0, 0, :]])
    assert torch.all(torch.eq(targ, x_time))


def test_slr_con_fixed(composer):
    model_name = model_str(model_names.CON, byr=False)
    feats = load_featnames(SLR_PREFIX, model_names.CON)['x_fixed']
    targ = torch.arange(len(feats)).float()
    sources = get_offer_fixed(composer)
    x_fixed, _ = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                             recurrent=False, size=1)
    assert torch.all(torch.eq(targ, x_fixed))


def test_slr_con_time(composer):
    model_name = model_str(model_names.CON, byr=False)
    feats = load_featnames(SLR_PREFIX, model_names.CON)['x_time']
    targ = torch.arange(len(feats)).float()
    sources = slr_early_offer_sources()
    sources[L_OUTCOMES_MAP] = torch.tensor([-1, -1, 50, 51, 52, 53, 54, 55, 56]).float()
    _, x_time = composer.build_input_vector(model_name, sources=sources, fixed=False,
                                            recurrent=True, size=1)
    # print(targ[~torch.eq(targ, x_time)[0, 0, :]])
    assert torch.all(torch.eq(targ, x_time))


def test_slr_msg_fixed(composer):
    model_name = model_str(model_names.MSG, byr=False)
    feats = load_featnames(SLR_PREFIX, model_names.MSG)['x_fixed']
    targ = torch.arange(len(feats)).float()
    sources = get_offer_fixed(composer)
    x_fixed, _ = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                             recurrent=False, size=1)
    assert torch.all(torch.eq(targ, x_fixed))


def test_slr_msg_time(composer):
    model_name = model_str(model_names.MSG, byr=False)
    feats = load_featnames(SLR_PREFIX, model_names.MSG)['x_time']
    targ = torch.arange(len(feats)).float()
    sources = get_offer_fixed(composer)
    start = 0
    tot = 2
    sources[OUTCOMES_MAP] = torch.tensor([0, 1, 10, 11, 12, 13, 14, 15]).float()
    start = tot
    tot += len(CLOCK_FEATS)
    sources[CLOCK_MAP] = torch.arange(start, tot).float()
    start = 13
    tot = start
    tot += len(TIME_FEATS)
    sources[DIFFS_MAP] = torch.arange(start, tot).float()
    sources[O_OUTCOMES_MAP] = torch.tensor([26, 27, 36, 37, 38, 39]).float()
    start = 28
    tot = start + len(CLOCK_FEATS)
    sources[O_CLOCK_MAP] = torch.arange(start, tot).float()
    start = 40
    tot = start + len(TIME_FEATS)
    sources[O_DIFFS_MAP] = torch.arange(start, tot).float()
    sources[L_OUTCOMES_MAP] = torch.tensor([-1, -1, -1, -1, -1, 53, 54, 55, 56]).float()
    start = 57
    tot = 59
    sources[TURN_IND_MAP] = torch.arange(start, tot).float()
    _, x_time = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                            recurrent=True, size=1)
    assert torch.all(torch.eq(targ, x_time))


def test_byr_delay_fixed(composer):
    model_name = model_str(model_names.DELAY, byr=True)
    feats = load_featnames(BYR_PREFIX, model_names.DELAY)['x_fixed']
    targ = torch.arange(len(feats)).float()
    sources = get_offer_fixed(composer)
    add_byr_delay_fixed(sources)
    add_delay_time(sources)
    x_fixed, _ = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                             recurrent=True, size=1)
    assert torch.all(torch.eq(targ, x_fixed))


def add_byr_delay_fixed(sources):
    sources[TURN_IND_MAP] = torch.tensor([-1, 197, 198]).float()
    sources[DAYS_THREAD_MAP] = torch.tensor([199]).float()
    start = 200
    tot = start + len(SLR_OUTCOMES)
    sources[O_OUTCOMES_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[O_TIME_MAP] = torch.arange(start, tot).float()
    sources[L_OUTCOMES_MAP] = torch.tensor([222, 223, 232, 233, 234, 235]).float()
    start = 224
    tot = start + len(CLOCK_FEATS)
    sources[L_CLOCK_MAP] = torch.arange(start, tot).float()
    start = 236
    tot = start + len(TIME_FEATS)
    sources[L_TIME_MAP] = torch.arange(start, tot).float()


def add_delay_time(sources):
    start = 0
    tot = start + len(CLOCK_FEATS)
    sources[CLOCK_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[DIFFS_MAP] = torch.arange(start, tot).float()
    sources[DUR_MAP] = torch.tensor([21]).float()
    sources[INT_REMAIN_MAP] = torch.tensor([22]).float()


def test_byr_delay_time(composer):
    model_name = model_str(model_names.DELAY, byr=True)
    feats = load_featnames(BYR_PREFIX, model_names.DELAY)['x_time']
    targ = torch.arange(len(feats)).float()
    sources = get_offer_fixed(composer)
    add_byr_delay_fixed(sources)
    add_delay_time(sources)
    _, x_time = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                            recurrent=True, size=1)
    assert torch.all(torch.eq(targ, x_time))


def test_slr_delay_fixed(composer):
    model_name = model_str(model_names.DELAY, byr=False)
    feats = load_featnames(SLR_PREFIX, model_names.DELAY)['x_fixed']
    targ = torch.arange(len(feats)).float()
    sources = get_offer_fixed(composer)
    add_slr_delay_fixed(sources)
    add_delay_time(sources)
    x_fixed, _ = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                             recurrent=False, size=1)
    assert torch.all(torch.eq(targ, x_fixed))


def test_slr_delay_time(composer):
    model_name = model_str(model_names.DELAY, byr=False)
    feats = load_featnames(SLR_PREFIX, model_names.DELAY)['x_time']
    targ = torch.arange(len(feats)).float()
    sources = get_offer_fixed(composer)
    add_slr_delay_fixed(sources)
    add_delay_time(sources)
    _, x_time = composer.build_input_vector(model_name, sources=sources, fixed=False,
                                            recurrent=True, size=1)
    assert torch.all(torch.eq(targ, x_time))


def add_slr_delay_fixed(sources):
    sources[TURN_IND_MAP] = torch.tensor([197, 198]).float()
    sources[DAYS_THREAD_MAP] = torch.tensor([199]).float()
    start = 200
    tot = start + len(BYR_OUTCOMES)
    sources[O_OUTCOMES_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[O_TIME_MAP] = torch.arange(start, tot).float()
    sources[L_OUTCOMES_MAP] = torch.tensor([219, 220, 229, 230, 231, 232, 233, 234, 235]).float()
    start = 221
    tot = start + len(CLOCK_FEATS)
    sources[L_CLOCK_MAP] = torch.arange(start, tot).float()
    start = 236
    tot = start + len(TIME_FEATS)
    sources[L_TIME_MAP] = torch.arange(start, tot).float()