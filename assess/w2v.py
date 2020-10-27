import pandas as pd
from agent.util import load_values, get_sale_norm
from utils import unpickle, topickle, load_data, load_feats
from assess.const import DELTA_SHAP
from constants import FEATS_DIR, PLOT_DIR
from featnames import META, LEAF, CNDTN, LOOKUP, TEST, X_OFFER, CON, LSTG


def create_df(s=None, leaf=None, tsne=None, reindex=True):
    if reindex:
        s = s.reindex(index=leaf.index, fill_value=False)
    df = s.rename('c').to_frame().join(leaf)
    df = df.groupby(LEAF)[['c']].mean()
    return df.join(tsne)


def main():
    # tsne coordinates
    tsne = unpickle(FEATS_DIR + 'tsne.pkl')

    # output dictionary
    d = {}

    # color by meta
    cats = tsne.index.get_level_values(META).unique()
    colors = pd.Series(range(len(cats)), index=cats, name='c')
    d['w2v_{}'.format(META)] = tsne.join(colors, on=META)

    # data
    data = load_data(part=TEST)
    leaf = load_feats('listings', lstgs=data[LOOKUP].index)[LEAF]

    # color by value
    vals = load_values(part=TEST, delta=DELTA_SHAP)
    mean_vals = vals.groupby(leaf).mean().rename('c')
    d['w2v_values'] = tsne.join(mean_vals, on=LEAF)

    # color by difference in value between new and used
    cndtn = data[LOOKUP][CNDTN]
    new = cndtn[cndtn == 1].index
    used = cndtn[cndtn == 7].index
    new_vals = vals.loc[new].groupby(leaf).mean().rename('c')
    used_vals = vals.loc[used].groupby(leaf).mean().rename('c')
    diff_vals = (new_vals - used_vals).dropna()
    d['w2v_{}'.format(CNDTN)] = diff_vals.to_frame().join(tsne)

    # color by Pr(sale)
    sale = (data[X_OFFER][CON] == 1).groupby(LSTG).max()
    d['w2v_sale'] = create_df(s=sale, leaf=leaf, tsne=tsne)

    # color by mean sale norm
    sale_norm = get_sale_norm(data[X_OFFER])
    d['w2v_norm'] = create_df(s=sale_norm, leaf=leaf, tsne=tsne, reindex=False)

    # save
    topickle(d, PLOT_DIR + 'w2v.pkl')


if __name__ == '__main__':
    main()
