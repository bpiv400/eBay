import pandas as pd
from agent.util import load_values
from utils import unpickle, topickle, load_data, load_feats
from assess.const import DELTA_ASSESS
from constants import FEATS_DIR, PLOT_DIR
from featnames import META, LEAF, LOOKUP, TEST


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
    vals = load_values(part=TEST, delta=DELTA_ASSESS)
    mean_vals = vals.groupby(leaf).mean().rename('c')
    d['w2v_values'] = tsne.join(mean_vals, on=LEAF)

    # save
    topickle(d, PLOT_DIR + 'w2v.pkl')


if __name__ == '__main__':
    main()
