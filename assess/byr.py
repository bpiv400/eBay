from agent.util import get_log_dir, get_valid_byr, load_values
from assess.util import discrete_pdf, discrete_cdf, get_sale_norm, norm_norm
from utils import load_data
from constants import TEST, RL_BYR
from featnames import CON, INDEX, THREAD, X_OFFER, X_THREAD, LOOKUP, START_PRICE, \
    DAYS_SINCE_LSTG, AUTO, LSTG, NORM

PART = TEST


def get_turn(s=None, turn=None, agent_threads=None):
    return s.xs(turn, level=INDEX).reindex(
        index=agent_threads).dropna().droplevel(THREAD)


def main():
    values = load_values(part=PART)

    # byr run
    byr_dir = get_log_dir(True) + 'run_full_min40_new_values/'
    data = load_data(part=PART, folder=byr_dir)
    norm_values = values / data[LOOKUP][START_PRICE]

    # restrict datea to valid
    data = get_valid_byr(data)
    valid = data[LOOKUP].index
    invalid = values.index.drop(valid)
    print('Valid listings: {0:.1f}%'.format(
        100 * len(valid) / len(values.index)))
    print('Avg norm value for valid [invalid] listings: {0:.1f}% [{1:.1f}%]'.format(
        100 * norm_values[valid].mean(), 100 * norm_values[invalid].mean()))

    # restrict values to valid
    norm_values = norm_values.reindex(index=valid)
    print('Share of valid norm values above 40%: {0:.1f}%'.format(
        100 * (norm_values > .4).mean()))

    # count and timing of sales
    sale = data[X_OFFER][data[X_OFFER][CON] == 1].index
    byr_agent = data[X_THREAD]['byr_agent'].reindex(index=sale)
    agent_sale = byr_agent[byr_agent].index
    other_sale = byr_agent[~byr_agent].index
    print('Share of valid listings resulting in sale: {0:.1f}%'.format(
        100 * len(sale) / len(valid)))
    print('Share of valid listings resulting in sale to agent: {0:.1f}%'.format(
        100 * len(agent_sale) / len(valid)))
    agent_sale_turn = agent_sale.get_level_values(INDEX)
    agent_sale_norm = data[X_OFFER].loc[agent_sale, NORM]
    for t in [2, 4, 6]:
        print('Share of agent sales in turn {0:d}: {1:.1f}%'.format(
            t, 100 * (agent_sale_turn == t).mean()))
        norm_t = 1 - agent_sale_norm.xs(t, level=INDEX).droplevel(THREAD)
        value_t = norm_values.loc[norm_t.index] - norm_t
        print('Average sale value in turn: {0:d}: {1:.1f}%'.format(
            t, 100 * value_t.mean()))

    # sale values
    sale_norm = get_sale_norm(offers=data[X_OFFER])
    sale_value = norm_values.loc[sale_norm.index] - sale_norm
    agent_sale_value = sale_value.loc[agent_sale.get_level_values(LSTG)]
    other_sale_value = sale_value.loc[other_sale.get_level_values(LSTG)]
    cdf_agent_sale_value = discrete_cdf(100 * agent_sale_value)
    cdf_other_sale_value = discrete_cdf(100 * other_sale_value)

    # when does it make the first offer
    agent_thread = data[X_THREAD][data[X_THREAD]['byr_agent']].index
    day = data[X_THREAD].loc[agent_thread, DAYS_SINCE_LSTG].droplevel(THREAD)
    print('Share of first offers on day 0: {0:.1f}%'.format(100 * (day < 1).mean()))

    # values for offer vs. walk
    idx_offer = agent_thread.droplevel(THREAD)
    idx_walk = valid.drop(idx_offer)
    cdf_offer = discrete_cdf(100 * norm_values[idx_offer])
    cdf_walk = discrete_cdf(100 * norm_values[idx_walk])

    # cdfs of sale values
    agent_sale_lstgs = agent_sale.droplevel([THREAD, INDEX])
    cdf_agent_sale = discrete_cdf(100 * norm_values[agent_sale_lstgs])

    # concessions by turn
    con, auto = dict(), dict()
    for t in range(1, 8):
        con[t] = get_turn(s=data[X_OFFER][CON],
                          turn=t,
                          agent_threads=agent_thread)
        con[t] *= 100
        con[t] = con[t].astype('uint8')
        if t > 1:
            if t % 2 == 0:
                prev_idx = con[t - 1][(con[t - 1] > 0) & (con[t - 1] < 100)].index
                con[t] = con[t].reindex(index=prev_idx, fill_value=0)
                auto[t] = get_turn(s=data[X_OFFER][AUTO],
                                   turn=t,
                                   agent_threads=agent_thread)
                auto[t] = auto[t].reindex(index=prev_idx, fill_value=False)

    # counterfactual for turn 7
    slr6 = data[X_OFFER][NORM].xs(6, level=INDEX)
    byr7_norm = 1 - slr6.reindex(index=agent_thread).dropna().droplevel(THREAD)
    byr7_value = norm_values.loc[byr7_norm.index] - byr7_norm
    cdf_byr7_value = discrete_cdf(100 * byr7_value)

    # counterfactual for turn 5
    con5 = data[X_OFFER][CON].xs(5, level=INDEX).reindex(agent_thread).dropna()
    norm4 = 1 - data[X_OFFER][NORM].xs(4, level=INDEX).reindex(agent_thread).dropna()
    norm5 = data[X_OFFER][NORM].xs(5, level=INDEX).reindex(agent_thread).dropna()
    high5 = (con5 == .4).astype(bool)
    high5_idx = high5[high5].index
    high5_norm4 = norm4[high5_idx].droplevel(THREAD)
    high5_acc5_value = norm_values.loc[high5_norm4.index] - high5_norm4
    print('Profitable accepts at turn 5: {0:.1f}%'.format(
        (high5_acc5_value > 0).mean()))
    high5_norm5 = norm5[high5_idx].droplevel(THREAD)
    high5_acc6_value = norm_values.loc[high5_norm5.index] - high5_norm5

    # when does agent make a large concession in turn 3
    agent_offers = data[X_OFFER][[NORM, AUTO]].unstack().loc[agent_thread].stack()
    norm_hat = norm_norm(agent_offers)


    # distribution of seller counters
    cdf_con2 = discrete_cdf(con[2])
    cdf_con4 = discrete_cdf(con[4])
    cdf_con6 = discrete_cdf(con[6])

    # auto vs. manual reject -- very similar
    con3_rej2 = discrete_pdf()
    con3_auto2 = discrete_pdf(con[3].loc[(con[1] == 40) & auto[2]])
    con3_man2 = discrete_pdf(con[3].loc[(con[1] == 40) & ~auto[2]])


if __name__ == 'main':
    main()
