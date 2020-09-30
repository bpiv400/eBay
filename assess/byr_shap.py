import pandas as pd
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
