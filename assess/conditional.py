from compress_pickle import dump
from utils import load_file
from inputs.discrim import load_threads_offers
from constants import TEST, PLOT_DIR, SIM, OBS, IDX, SLR
from featnames import META, START_PRICE, CON, NORM


def sale_price_rate(df, idx):
    norm = df.loc[df[CON] == 1, NORM]
    slr = norm.index.isin(IDX[SLR], level='index')
    norm.loc[slr] = 1 - norm.loc[slr]
    return norm.mean(), len(norm) / len(idx)


def create_outputs(obs, sim, idx):
    p = {SIM: dict(), OBS: dict()}

    # average normalizad sale price and sale rate
    p[SIM]['price'], p[SIM]['sale'] = \
        sale_price_rate(sim['offers'], idx)
    p[OBS]['price'], p[OBS]['sale'] = \
        sale_price_rate(obs['offers'], idx)

    return p


def partition_d(d, idx):
    return {k: v.reindex(index=idx, level='lstg') for k, v in d.items()}


def main():
    # observed and simulated outcomes
    threads_obs, offers_obs = load_threads_offers(sim=False)
    threads_sim, offers_sim = load_threads_offers(sim=True)

    # lookup file
    lookup = load_file(TEST, 'lookup')

    # conditional distributions split by category or interval
    for split_feat in SPLIT_VALS.keys():
        ps = []  # list of dictionaries

        # create indices
        s, indices = lookup[split_feat], []
        if split_feat == META:
            for i in range(len(SPLIT_VALS[split_feat])):
                mask = s == SPLIT_VALS[split_feat][i]
                indices.append(s[mask].index)
        elif split_feat == START_PRICE:
            for i in range(len(SPLIT_VALS[split_feat]) - 1):
                lower = s >= SPLIT_VALS[split_feat][i]
                upper = s < SPLIT_VALS[split_feat][i+1]
                indices.append(s[lower & upper].index)
        else:
            RuntimeError('Invalid entry: {}'.format(split_feat))

        for i in range(len(indices)):
            idx = indices[i]
            obs_i = partition_d(obs, idx)
            sim_i = partition_d(sim, idx)
            ps.append(create_outputs(obs_i, sim_i, idx))

        # save
        dump(ps, PLOT_DIR + 'p_{}.pkl'.format(split_feat))


if __name__ == '__main__':
    main()
