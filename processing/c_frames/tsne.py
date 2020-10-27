import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from utils import unpickle, topickle, load_file, load_feats
from assess.const import META_LABELS
from constants import FEATS_DIR
from featnames import BYR, SLR, LEAF, LOOKUP, META, TEST


def get_meta_labels():
    cats = load_file(TEST, LOOKUP)[[META, LEAF]]
    meta = cats.groupby(LEAF)[META].first()
    labels = META_LABELS.values
    labels = [label.split(' & ')[0] for label in labels]
    labels = [label.split('/')[0] for label in labels]
    toReplace = {'Everything Else': '',
                 'Travel': '',
                 'Pet Supplies': '',
                 'Sports Mem Cards': 'Sports Mem.',
                 'Clothing Shoes': 'Clothing',
                 'Consumer Electronics': 'Electronics',
                 'Musical Instruments': 'Musical Instr.',
                 'Entertainment Memorabilia': 'Entertainment Mem.'}
    for old, new in toReplace.items():
        labels = [new if label == old else label for label in labels]
    labels = pd.Series(labels, name='label', index=META_LABELS.index)
    df = meta.to_frame().join(labels, on=META)
    return df


def main():
    leaf = load_feats('listings')[LEAF]
    leafs = np.sort(leaf.unique())

    # w2v features
    tojoin = []
    for role in [BYR, SLR]:
        path = FEATS_DIR + 'w2v_{}.pkl'.format(role)
        tojoin.append(unpickle(path))
    w2v = pd.concat(tojoin, axis=1).reindex(index=leafs)
    w2v[w2v.isna()] = 0.

    # reduce using t-SNE
    v = TSNE(n_jobs=-1, random_state=0).fit_transform(w2v.values)

    # put in dataframe
    tsne = pd.DataFrame(v, columns=['x', 'y'], index=w2v.index)
    tsne['s'] = leaf.groupby(leaf).count()

    # add meta
    tsne = get_meta_labels().join(tsne)
    tsne = tsne.reset_index().set_index([META, LEAF]).sort_index()

    # save
    topickle(tsne, FEATS_DIR + 'tsne.pkl')


if __name__ == '__main__':
    main()
