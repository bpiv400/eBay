import pytest
import torch
from rlenv.Composer import Composer
from rlenv.model_names import *
from rlenv.env_consts import *
from rlenv.env_utils import *

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


def byr_early_offer_sources():
    sources = dict()
    sources[OUTCOMES_MAP] = torch.arange(0, len(BYR_OUTCOMES))
    tot = len(OFFER_CLOCK_FEATS) + 1
    sources[CLOCK_MAP] = torch.arange(1, len(OFFER_CLOCK_FEATS) + 1).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[TIME_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[DIFFS_MAP] = torch.arange(start, tot).float()
    sources[O_OUTCOMES_MAP] = torch.tensor([33, 26, 27, 29, 30, 31, 32, 34, 28, 35]).float()
    start = 36
    tot = start + len(OFFER_CLOCK_FEATS)
    sources[O_CLOCK_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[O_TIME_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[O_DIFFS_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(BYR_OUTCOMES)
    sources[L_OUTCOMES_MAP] = torch.arange(start - 1, tot).float()
    start = 67
    tot = 70
    sources[TURN_IND_MAP] = torch.arange(start, tot).float()
    sources[LSTG_MAP] = torch.arange(2, len(LSTG_COLS) + 2).float()
    sources[BYR_ATTR_MAP] = torch.arange(2).float()
    return sources


def slr_early_offer_sources():
    sources = dict()
    sources[OUTCOMES_MAP] = torch.arange(0, len(SLR_OUTCOMES))
    tot = len(OFFER_CLOCK_FEATS) + 1
    sources[CLOCK_MAP] = torch.arange(1, len(OFFER_CLOCK_FEATS) + 1).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[TIME_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[DIFFS_MAP] = torch.arange(start, tot).float()
    sources[O_OUTCOMES_MAP] = torch.tensor([32, 26, 27, 28, 29, 30, 31]).float()
    start = 33
    tot = start + len(OFFER_CLOCK_FEATS)
    sources[O_CLOCK_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[O_TIME_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[O_DIFFS_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(SLR_OUTCOMES)
    sources[L_OUTCOMES_MAP] = torch.arange(start - 1, tot).float()
    sources[L_OUTCOMES_MAP][3:8] = torch.arange(start + 3, start + 8).float()
    sources[L_OUTCOMES_MAP][8] = start + 2
    start = 67
    tot = 69
    sources[TURN_IND_MAP] = torch.arange(start, tot).float()
    sources[LSTG_MAP] = torch.arange(2, len(LSTG_COLS) + 2).float()
    sources[BYR_ATTR_MAP] = torch.arange(2).float()
    return sources


def test_loc_input_fixed(composer):
    model_name = LOC
    loc_feats = load_featnames(ARRIVAL_PREFIX, model_name)['x_fixed']
    targ = torch.arange(len(loc_feats)).float()
    sources = {
        LSTG_MAP: torch.arange(len(LSTG_COLS)).float(),
        CLOCK_MAP: torch.arange(len(LSTG_COLS), len(LSTG_COLS) + len(FF_CLOCK_FEATS)).float()
    }
    x_fixed, _ = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                             recurrent=False, size=1)
    assert torch.all(torch.eq(targ, x_fixed))


def test_hist_input_us(composer):
    model_name = HIST
    feats = load_featnames(ARRIVAL_PREFIX, model_name)['x_fixed']
    targ = torch.arange(len(feats)).float()
    targ[132] = 1
    sources = {
        LSTG_MAP: torch.arange(len(LSTG_COLS)).float(),
        CLOCK_MAP: torch.arange(len(LSTG_COLS), len(LSTG_COLS) + len(FF_CLOCK_FEATS)).float()
    }
    x_fixed = composer.build_hist_input(sources=sources, us=True, foreign=False)
    assert torch.all(torch.eq(targ, x_fixed))


def test_hist_input_foreign(composer):
    model_name = HIST
    feats = load_featnames(ARRIVAL_PREFIX, model_name)['x_fixed']
    targ = torch.arange(len(feats)).float()
    targ[132] = 0
    sources = {
        LSTG_MAP: torch.arange(len(LSTG_COLS)).float(),
        CLOCK_MAP: torch.arange(len(LSTG_COLS), len(LSTG_COLS) + len(FF_CLOCK_FEATS)).float()
    }
    x_fixed = composer.build_hist_input(sources=sources, us=False, foreign=True)
    assert torch.all(torch.eq(targ, x_fixed))


def test_hist_input_both(composer):
    model_name = HIST
    feats = load_featnames(ARRIVAL_PREFIX, model_name)['x_fixed']
    part_targ = torch.arange(len(feats) - 1).float()
    targ = torch.zeros(2, len(feats)).float()
    targ[:, 0:len(feats) - 1] = part_targ
    targ[0, len(feats) - 1] = 0
    targ[1, len(feats) - 1] = 1

    sources = {
        LSTG_MAP: torch.arange(len(LSTG_COLS)).float(),
        CLOCK_MAP: torch.arange(len(LSTG_COLS), len(LSTG_COLS) + len(FF_CLOCK_FEATS)).float()
    }
    x_fixed = composer.build_hist_input(sources=sources, us=True, foreign=True)
    assert torch.all(torch.eq(targ, x_fixed))


def test_sec_input_one(composer):
    model_name = SEC
    feats = load_featnames(ARRIVAL_PREFIX, model_name)['x_fixed']
    targ = torch.arange(len(feats)).float()
    sources = {
        LSTG_MAP: torch.arange(len(LSTG_COLS)).float(),
        CLOCK_MAP: torch.arange(len(LSTG_COLS), len(LSTG_COLS) + len(FF_CLOCK_FEATS)).float(),
        BYR_US_MAP: torch.tensor(132).float(),
        BYR_HIST_MAP: torch.tensor(133).float()
    }
    x_fixed, _ = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                             recurrent=False, size=1)
    assert torch.all(torch.eq(targ, x_fixed))


def test_sec_input_mult(composer):
    model_name = SEC
    feats = load_featnames(ARRIVAL_PREFIX, model_name)['x_fixed']
    part_targ = torch.arange(len(feats) - 2).float()
    targ = torch.zeros(2, len(feats)).float()
    # clock and lstg feats
    targ[:,0:len(feats) - 2] = part_targ
    # byr us
    targ[0, len(feats) - 2] = 0
    targ[1, len(feats) - 2] = 1
    # exp
    targ[0, len(feats) - 1] = 15
    targ[1, len(feats) - 1] = 4
    sources = {
        LSTG_MAP: torch.arange(len(LSTG_COLS)).float(),
        CLOCK_MAP: torch.arange(len(LSTG_COLS), len(LSTG_COLS) + len(FF_CLOCK_FEATS)).float(),
        BYR_US_MAP: torch.tensor([0, 1]).float().unsqueeze(1),
        BYR_HIST_MAP: torch.tensor([15, 4]).float().unsqueeze(1)
    }
    x_fixed, _ = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                             recurrent=False, size=2)
    assert torch.all(torch.eq(targ, x_fixed))


def test_byr_acc_fixed(composer):
    model_name = model_str(ACC, byr=True)
    feats = load_featnames(BYR_PREFIX, ACC)['x_fixed']
    targ = torch.arange(len(feats)).float()
    sources = {
        LSTG_MAP: torch.arange(2, len(LSTG_COLS) + 2).float(),
        BYR_ATTR_MAP: torch.arange(2).float()
    }
    x_fixed, _ = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                             recurrent=False, size=1)
    assert torch.all(torch.eq(targ, x_fixed))


def test_byr_acc_time(composer):
    model_name = model_str(ACC, byr=True)
    feats = load_featnames(BYR_PREFIX, ACC)['x_time']
    targ = torch.arange(len(feats)).float()
    sources = byr_early_offer_sources()
    _, x_time = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                            recurrent=True, size=1)
    assert torch.all(torch.eq(targ, x_time))


def test_byr_rej_fixed(composer):
    model_name = model_str(REJ, byr=True)
    feats = load_featnames(BYR_PREFIX, REJ)['x_fixed']
    targ = torch.arange(len(feats)).float()
    sources = {
        LSTG_MAP: torch.arange(2, len(LSTG_COLS) + 2).float(),
        BYR_ATTR_MAP: torch.arange(2).float()
    }
    x_fixed, _ = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                             recurrent=False, size=1)
    assert torch.all(torch.eq(targ, x_fixed))


def test_byr_rej_time(composer):
    model_name = model_str(REJ, byr=True)
    feats = load_featnames(BYR_PREFIX, REJ)['x_time']
    targ = torch.arange(len(feats)).float()
    sources = byr_early_offer_sources()
    _, x_time = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                            recurrent=True, size=1)
    assert torch.all(torch.eq(targ, x_time))


def test_byr_con_fixed(composer):
    model_name = model_str(CON, byr=True)
    feats = load_featnames(BYR_PREFIX, CON)['x_fixed']
    targ = torch.arange(len(feats)).float()
    sources = {
        LSTG_MAP: torch.arange(2, len(LSTG_COLS) + 2).float(),
        BYR_ATTR_MAP: torch.arange(2).float()
    }
    x_fixed, _ = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                             recurrent=False, size=1)
    assert torch.all(torch.eq(targ, x_fixed))


def test_byr_con_time(composer):
    model_name = model_str(CON, byr=True)
    feats = load_featnames(BYR_PREFIX, CON)['x_time']
    targ = torch.arange(len(feats)).float()
    sources = byr_early_offer_sources()
    _, x_time = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                            recurrent=True, size=1)
    assert torch.all(torch.eq(targ, x_time))

def test_byr_msg_fixed(composer):
    model_name = model_str(MSG, byr=True)
    feats = load_featnames(BYR_PREFIX, MSG)['x_fixed']
    targ = torch.arange(len(feats)).float()
    sources = {
        LSTG_MAP: torch.arange(2, len(LSTG_COLS) + 2).float(),
        BYR_ATTR_MAP: torch.arange(2).float()
    }
    x_fixed, _ = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                             recurrent=False, size=1)
    assert torch.all(torch.eq(targ, x_fixed))

def test_byr_msg_time(composer):
    model_name = model_str(MSG, byr=True)
    feats = load_featnames(BYR_PREFIX, MSG)['x_time']
    targ = torch.arange(len(feats)).float()
    # make sources
    sources = dict()
    sources[OUTCOMES_MAP] = torch.arange(-1, len(BYR_OUTCOMES)).float()
    sources[OUTCOMES_MAP][0] = 3
    tot = len(OFFER_CLOCK_FEATS) + 4
    sources[CLOCK_MAP] = torch.arange(4, len(OFFER_CLOCK_FEATS) + 4).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[TIME_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[DIFFS_MAP] = torch.arange(start, tot).float()
    sources[O_OUTCOMES_MAP] = torch.tensor([36, 29, 30, 32, 33, 34, 35, 37, 31, 38]).float()
    start = 39
    tot = start + len(OFFER_CLOCK_FEATS)
    sources[O_CLOCK_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[O_TIME_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[O_DIFFS_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(BYR_OUTCOMES)
    sources[L_OUTCOMES_MAP] = torch.arange(start - 4, tot - 4).float()
    start = 67
    tot = 70
    sources[TURN_IND_MAP] = torch.arange(start, tot).float()
    sources[LSTG_MAP] = torch.arange(2, len(LSTG_COLS) + 2).float()
    sources[BYR_ATTR_MAP] = torch.arange(2).float()

    _, x_time = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                            recurrent=True, size=1)
    assert torch.all(torch.eq(targ, x_time))


def test_byr_rnd_fixed(composer):
    model_name = model_str(RND, byr=True)
    feats = load_featnames(BYR_PREFIX, RND)['x_fixed']
    targ = torch.arange(len(feats)).float()
    sources = {
        LSTG_MAP: torch.arange(2, len(LSTG_COLS) + 2).float(),
        BYR_ATTR_MAP: torch.arange(2).float()
    }
    x_fixed, _ = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                             recurrent=False, size=1)
    assert torch.all(torch.eq(targ, x_fixed))


def test_byr_rnd_time(composer):
    model_name = model_str(RND, byr=True)
    feats = load_featnames(BYR_PREFIX, RND)['x_time']
    targ = torch.arange(len(feats)).float()
    # make sources
    sources = dict()
    sources[OUTCOMES_MAP] = torch.arange(-1, len(BYR_OUTCOMES)).float()
    sources[OUTCOMES_MAP][6] = 3
    sources[OUTCOMES_MAP][0] = 4
    tot = len(OFFER_CLOCK_FEATS) + 5
    sources[CLOCK_MAP] = torch.arange(5, len(OFFER_CLOCK_FEATS) + 5).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[TIME_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[DIFFS_MAP] = torch.arange(start, tot).float()
    sources[O_OUTCOMES_MAP] = torch.tensor([37, 30, 31, 33, 34, 35, 36, 38, 32, 39]).float()
    start = 40
    tot = start + len(OFFER_CLOCK_FEATS)
    sources[O_CLOCK_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[O_TIME_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[O_DIFFS_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(BYR_OUTCOMES)
    sources[L_OUTCOMES_MAP] = torch.arange(start - 4, tot - 4).float()
    start = 67
    tot = 70
    sources[TURN_IND_MAP] = torch.arange(start, tot).float()
    sources[LSTG_MAP] = torch.arange(2, len(LSTG_COLS) + 2).float()
    sources[BYR_ATTR_MAP] = torch.arange(2).float()

    _, x_time = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                            recurrent=True, size=1)
    assert torch.all(torch.eq(targ, x_time))

def test_byr_nines_fixed(composer):
    model_name = model_str(NINE, byr=True)
    feats = load_featnames(BYR_PREFIX, NINE)['x_fixed']
    targ = torch.arange(len(feats)).float()
    sources = {
        LSTG_MAP: torch.arange(2, len(LSTG_COLS) + 2).float(),
        BYR_ATTR_MAP: torch.arange(2).float()
    }
    x_fixed, _ = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                             recurrent=False, size=1)
    assert torch.all(torch.eq(targ, x_fixed))


def test_byr_nines_time(composer):
    model_name = model_str(NINE, byr=True)
    feats = load_featnames(BYR_PREFIX, NINE)['x_time']
    targ = torch.arange(len(feats)).float()
    # make sources
    sources = dict()
    sources[OUTCOMES_MAP] = torch.arange(-1, len(BYR_OUTCOMES)).float()
    sources[OUTCOMES_MAP][6] = 4
    sources[OUTCOMES_MAP][0] = 5
    tot = len(OFFER_CLOCK_FEATS) + 6
    sources[CLOCK_MAP] = torch.arange(6, len(OFFER_CLOCK_FEATS) + 6).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[TIME_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[DIFFS_MAP] = torch.arange(start, tot).float()
    sources[O_OUTCOMES_MAP] = torch.tensor([38, 31, 32, 34, 35, 36, 37, 39, 33, 40]).float()
    start = 41
    tot = start + len(OFFER_CLOCK_FEATS)
    sources[O_CLOCK_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[O_TIME_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[O_DIFFS_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(BYR_OUTCOMES)
    sources[L_OUTCOMES_MAP] = torch.arange(start - 5, tot - 5).float()
    start = 67
    tot = 70
    sources[TURN_IND_MAP] = torch.arange(start, tot).float()
    sources[LSTG_MAP] = torch.arange(2, len(LSTG_COLS) + 2).float()
    sources[BYR_ATTR_MAP] = torch.arange(2).float()

    _, x_time = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                            recurrent=True, size=1)
    assert torch.all(torch.eq(targ, x_time))


def test_slr_acc_fixed(composer):
    model_name = model_str(ACC, byr=False)
    feats = load_featnames(SLR_PREFIX, ACC)['x_fixed']
    targ = torch.arange(len(feats)).float()
    sources = {
        LSTG_MAP: torch.arange(2, len(LSTG_COLS) + 2).float(),
        BYR_ATTR_MAP: torch.arange(2).float()
    }
    x_fixed, _ = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                             recurrent=False, size=1)
    assert torch.all(torch.eq(targ, x_fixed))


def test_slr_acc_time(composer):
    model_name = model_str(ACC, byr=False)
    feats = load_featnames(SLR_PREFIX, ACC)['x_time']
    targ = torch.arange(len(feats)).float()
    sources = slr_early_offer_sources()
    _, x_time = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                            recurrent=True, size=1)
    assert torch.all(torch.eq(targ, x_time))


def test_slr_rej_fixed(composer):
    model_name = model_str(REJ, byr=False)
    feats = load_featnames(SLR_PREFIX, REJ)['x_fixed']
    targ = torch.arange(len(feats)).float()
    sources = {
        LSTG_MAP: torch.arange(2, len(LSTG_COLS) + 2).float(),
        BYR_ATTR_MAP: torch.arange(2).float()
    }
    x_fixed, _ = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                             recurrent=False, size=1)
    assert torch.all(torch.eq(targ, x_fixed))


def test_slr_rej_time(composer):
    model_name = model_str(REJ, byr=False)
    feats = load_featnames(SLR_PREFIX, REJ)['x_time']
    targ = torch.arange(len(feats)).float()
    sources = slr_early_offer_sources()
    _, x_time = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                            recurrent=True, size=1)
    assert torch.all(torch.eq(targ, x_time))


def test_slr_con_fixed(composer):
    model_name = model_str(CON, byr=False)
    feats = load_featnames(SLR_PREFIX, CON)['x_fixed']
    targ = torch.arange(len(feats)).float()
    sources = {
        LSTG_MAP: torch.arange(2, len(LSTG_COLS) + 2).float(),
        BYR_ATTR_MAP: torch.arange(2).float()
    }
    x_fixed, _ = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                             recurrent=False, size=1)
    assert torch.all(torch.eq(targ, x_fixed))


def test_slr_con_time(composer):
    model_name = model_str(CON, byr=False)
    feats = load_featnames(SLR_PREFIX, CON)['x_time']
    targ = torch.arange(len(feats)).float()
    sources = slr_early_offer_sources()
    _, x_time = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                            recurrent=True, size=1)
    assert torch.all(torch.eq(targ, x_time))


def test_slr_msg_fixed(composer):
    model_name = model_str(MSG, byr=False)
    feats = load_featnames(SLR_PREFIX, MSG)['x_fixed']
    targ = torch.arange(len(feats)).float()
    sources = {
        LSTG_MAP: torch.arange(2, len(LSTG_COLS) + 2).float(),
        BYR_ATTR_MAP: torch.arange(2).float()
    }
    x_fixed, _ = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                             recurrent=False, size=1)
    assert torch.all(torch.eq(targ, x_fixed))


def test_slr_msg_time(composer):
    model_name = model_str(MSG, byr=False)
    feats = load_featnames(SLR_PREFIX, MSG)['x_time']
    targ = torch.arange(len(feats)).float()
    sources = dict()
    sources[OUTCOMES_MAP] = torch.arange(-1, len(SLR_OUTCOMES)).float()
    sources[OUTCOMES_MAP][0] = 3
    tot = len(OFFER_CLOCK_FEATS) + 4
    sources[CLOCK_MAP] = torch.arange(4, len(OFFER_CLOCK_FEATS) + 4).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[TIME_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[DIFFS_MAP] = torch.arange(start, tot).float()
    sources[O_OUTCOMES_MAP] = torch.tensor([35, 29, 30, 31, 32, 33, 34]).float()
    start = 36
    tot = start + len(OFFER_CLOCK_FEATS)
    sources[O_CLOCK_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[O_TIME_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[O_DIFFS_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(SLR_OUTCOMES) # start = 61
    sources[L_OUTCOMES_MAP] = torch.arange(start - 3, tot - 3).float()
    sources[L_OUTCOMES_MAP][8] = start
    sources[L_OUTCOMES_MAP][9] = 66
    start = 67
    tot = 69
    sources[TURN_IND_MAP] = torch.arange(start, tot).float()
    sources[LSTG_MAP] = torch.arange(2, len(LSTG_COLS) + 2).float()
    sources[BYR_ATTR_MAP] = torch.arange(2).float()
    _, x_time = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                            recurrent=True, size=1)
    assert torch.all(torch.eq(targ, x_time))


def test_slr_round_fixed(composer):
    model_name = model_str(RND, byr=False)
    feats = load_featnames(SLR_PREFIX, RND)['x_fixed']
    targ = torch.arange(len(feats)).float()
    sources = {
        LSTG_MAP: torch.arange(2, len(LSTG_COLS) + 2).float(),
        BYR_ATTR_MAP: torch.arange(2).float()
    }
    x_fixed, _ = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                             recurrent=False, size=1)
    assert torch.all(torch.eq(targ, x_fixed))


def test_slr_round_time(composer):
    model_name = model_str(RND, byr=False)
    feats = load_featnames(SLR_PREFIX, RND)['x_time']
    targ = torch.arange(len(feats)).float()
    sources = dict()
    sources[OUTCOMES_MAP] = torch.arange(-1, len(SLR_OUTCOMES)).float()
    sources[OUTCOMES_MAP][6] = 3
    sources[OUTCOMES_MAP][0] = 4
    tot = len(OFFER_CLOCK_FEATS) + 5
    sources[CLOCK_MAP] = torch.arange(5, len(OFFER_CLOCK_FEATS) + 5).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[TIME_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[DIFFS_MAP] = torch.arange(start, tot).float()
    sources[O_OUTCOMES_MAP] = torch.tensor([36, 30, 31, 32, 33, 34, 35]).float()
    start = 37
    tot = start + len(OFFER_CLOCK_FEATS)
    sources[O_CLOCK_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[O_TIME_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[O_DIFFS_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(SLR_OUTCOMES)  # start = 62
    sources[L_OUTCOMES_MAP] = torch.tensor([-1, -1, -1, -1, 63, 64, -1, 65, 62, 66]).float()
    start = 67
    tot = 69
    sources[TURN_IND_MAP] = torch.arange(start, tot).float()
    sources[LSTG_MAP] = torch.arange(2, len(LSTG_COLS) + 2).float()
    sources[BYR_ATTR_MAP] = torch.arange(2).float()
    _, x_time = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                            recurrent=True, size=1)
    assert torch.all(torch.eq(targ, x_time))


def test_slr_nine_time(composer):
    model_name = model_str(NINE, byr=False)
    feats = load_featnames(SLR_PREFIX, NINE)['x_time']
    targ = torch.arange(len(feats)).float()
    sources = dict()
    sources[OUTCOMES_MAP] = torch.arange(-1, len(SLR_OUTCOMES)).float()
    sources[OUTCOMES_MAP][6] = 4
    sources[OUTCOMES_MAP][0] = 5
    tot = len(OFFER_CLOCK_FEATS) + 6
    sources[CLOCK_MAP] = torch.arange(6, len(OFFER_CLOCK_FEATS) + 6).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[TIME_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[DIFFS_MAP] = torch.arange(start, tot).float()
    sources[O_OUTCOMES_MAP] = torch.tensor([37, 31, 32, 33, 34, 35, 36]).float()
    start = 38
    tot = start + len(OFFER_CLOCK_FEATS)
    sources[O_CLOCK_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[O_TIME_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    sources[O_DIFFS_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(SLR_OUTCOMES)  # start = 63
    sources[L_OUTCOMES_MAP] = torch.tensor([-1, -1, -1, -1, -1, 64, -1, 65, 63, 66]).float()
    start = 67
    tot = 69
    sources[TURN_IND_MAP] = torch.arange(start, tot).float()
    sources[LSTG_MAP] = torch.arange(2, len(LSTG_COLS) + 2).float()
    sources[BYR_ATTR_MAP] = torch.arange(2).float()
    _, x_time = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                            recurrent=True, size=1)
    assert torch.all(torch.eq(targ, x_time))

def test_slr_nine_fixed(composer):
    model_name = model_str(NINE, byr=False)
    feats = load_featnames(SLR_PREFIX, NINE)['x_fixed']
    targ = torch.arange(len(feats)).float()
    sources = {
        LSTG_MAP: torch.arange(2, len(LSTG_COLS) + 2).float(),
        BYR_ATTR_MAP: torch.arange(2).float()
    }
    x_fixed, _ = composer.build_input_vector(model_name, sources=sources, fixed=True,
                                             recurrent=False, size=1)
    assert torch.all(torch.eq(targ, x_fixed))


def test_byr_delay_fixed(composer):
    model_name = model_str(DELAY, byr=True)
    feats = load_featnames(BYR_PREFIX, DELAY)['x_fixed']
    targ = torch.arange(len(feats)).float()
    sources = dict()
    start = 0
    tot = len(BYR_TURN_INDS)
    sources[TURN_IND_MAP] = torch.arange(start, tot).float()
    start = 3
    tot += len(LSTG_COLS)
    sources[LSTG_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(BYR_ATTRS)
    sources[BYR_ATTR_MAP] = torch.arange(start, tot).float()
    sources[O_OUTCOMES_MAP] = torch.tensor([136, 129, 130, 132, 133, 134, 135, 137, 131, 138]).float()
    start = tot + len(SLR_OUTCOMES)
    tot = start + len(OFFER_CLOCK_FEATS)
    sources[O_CLOCK_MAP] = torch.arange(start, tot).float()
    start = tot
    tot += len(TIME_FEATS)
    
def test_byr_delay_time(composer):
    pass

def test_slr_delay_fixed(composer):
    pass

def test_slr_delay_time(composer):
    pass