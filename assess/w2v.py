import pandas as pd
from assess.util import save_dict
from utils import unpickle
from constants import FEATS_DIR
from featnames import META, LEAF


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

    # save
    save_dict(d, 'w2v')


if __name__ == '__main__':
    main()
