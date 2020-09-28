import pandas as pd
from inputs.policy_byr import append_arrival_delays, reshape_offers, construct_x
from agent.util import get_log_dir, get_byr_valid
from utils import load_data
from constants import TEST, IDX
from featnames import BYR, INDEX, THREAD, LSTG, X_OFFER, X_THREAD, LOOKUP, CLOCK, \
    START_TIME

# load byr data
byr_dir = get_log_dir(True) + 'run_full_delta_0.9/'
data = load_data(part=TEST, folder=byr_dir)
data = get_byr_valid(data)

# subset to agent threads
agent_threads = data[X_THREAD][data[X_THREAD]['byr_agent']].drop(
    'byr_agent', axis=1)

agent_offers = pd.DataFrame(index=agent_threads.index).join(data[X_OFFER])
agent_clock = data[CLOCK].reindex(index=agent_offers.index)
agent_delays = data['delays'].to_frame().assign(index=1).join(
    agent_threads.reset_index(THREAD)[THREAD])
agent_delays[THREAD] = agent_delays[THREAD].fillna(0).astype('uint8')
agent_delays = agent_delays.set_index([THREAD, INDEX], append=True).reorder_levels(
    [LSTG, THREAD, INDEX, 'day']).squeeze()

# processing from inputs
agent_clock, idx0, idx1 = append_arrival_delays(
    clock=agent_clock,
    lstg_start=data[LOOKUP][START_TIME],
    delays=agent_delays
)
agent_offers = reshape_offers(offers=agent_offers,
                              clock=agent_clock,
                              idx0=idx0,
                              idx1=idx1)
idx = agent_clock[agent_clock.index.isin(IDX[BYR], level=INDEX)].index
x = construct_x(idx=idx,
                threads=agent_threads,
                clock=agent_clock,
                lookup=data[LOOKUP],
                offers=agent_offers)
