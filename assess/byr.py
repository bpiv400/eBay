from agent.util import find_best_run, get_log_dir, get_valid_byr, load_values
from assess.util import discrete_pdf
from utils import load_data
from constants import TEST
from featnames import CON, INDEX, THREAD, X_OFFER, X_THREAD, LOOKUP, START_PRICE, \
    DAYS_SINCE_LSTG, AUTO


def get_turn(s=None, turn=None, agent_threads=None):
    return s.xs(turn, level=INDEX).reindex(
        index=agent_threads).dropna().droplevel(THREAD)


def main():
    values = load_values(part=TEST)

    # byr run
    # best_byr_dir = find_best_run(byr=True, verbose=False)
    best_byr_dir = get_log_dir(True) + 'run_full_nofees_min40/'
    data = load_data(part=TEST, folder=best_byr_dir)
    norm_values = values / data[LOOKUP][START_PRICE]

    # restrict to valid
    data = get_valid_byr(data)
    print('Valid listings: {0:.1f}%'.format(
        100 * len(data[LOOKUP].index) / len(values.index)))
    print('Share of norm value above 40%: {0:.1f}%'.format(
        100 * (norm_values > .4).mean()))
    norm_values = norm_values.reindex(index=data[LOOKUP].index)
    print('Share of valid norm values above 40%: {0:.1f}%'.format(
        100 * (norm_values > .4).mean()))

    # concessions by turn
    agent_threads = data[X_THREAD][data[X_THREAD]['byr_agent']].index
    con, auto = dict(), dict()
    for t in range(1, 8):
        con[t] = get_turn(s=data[X_OFFER][CON],
                          turn=t,
                          agent_threads=agent_threads)
        if t > 1:  # TODO: allow for end of listing
            if t % 2 == 0:
                prev_idx = con[t-1][(con[t-1] > 0) & (con[t-1] < 1)].index
                con[t] = con[t].reindex(index=prev_idx, fill_value=0)

                auto[t] = get_turn(s=data[X_OFFER][AUTO],
                                   turn=t,
                                   agent_threads=agent_threads)
                auto[t] = auto[t].reindex(index=prev_idx, fill_value=False)

    # when does it make the first offer
    day = data[X_THREAD].loc[agent_threads, DAYS_SINCE_LSTG].droplevel(THREAD)
    time = day % 1
    day = day.astype('uint8')
    late = day > 0

    # low offer, auto vs. manual reject -- very similar
    low_auto = discrete_pdf(100 * con[3].loc[(con[1] == 0.4) & auto[2]])
    low_man = discrete_pdf(100 * con[3].loc[(con[1] == 0.4) & ~auto[2]])



if __name__ == 'main':
    main()
