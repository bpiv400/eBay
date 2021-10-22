import pandas as pd
from sklearn.manifold import TSNE
from analyze.util import save_dict
from utils import unpickle, load_feats
from paths import FEATS_DIR
from featnames import BYR, SLR, LEAF, META

NO_LABEL = ['Everything Else', 'Travel', 'Pet Supplies']
META_LABELS = {1: 'Collectibles',
               99: 'Everything Else',
               220: 'Toys',
               237: 'Dolls',
               260: 'Stamps',
               267: 'Books',
               281: 'Jewelry',
               293: 'Electronics',
               550: 'Art',
               619: 'Musical Instr.',
               625: 'Cameras',
               870: 'Pottery',
               888: 'Sporting Goods',
               1249: 'Video Games',
               1281: 'Pet Supplies',
               1305: 'Tickets',
               2984: 'Baby',
               3252: 'Travel',
               11116: 'Coins',
               11232: 'DVDs',
               11233: 'Music',
               11450: 'Clothing',
               11700: 'Home',
               12576: 'Business',
               14339: 'Crafts',
               15032: 'Cell Phones',
               20081: 'Antiques',
               26395: 'Health',
               45100: 'Entertainment Mem.',
               58058: 'Computers',
               64482: 'Sports Mem.'}


def create_df(s=None, leaf=None, tsne=None, reindex=True):
    if reindex:
        s = s.reindex(index=leaf.index, fill_value=False)
    df = s.rename('c').to_frame().join(leaf)
    df = df.groupby(LEAF)[['c']].mean()
    return df.join(tsne)


def do_tsne():
    cats = load_feats('listings')[[META, LEAF]]
    leaf_ct = cats.groupby(LEAF).count().squeeze().rename('count')

    # labels
    meta = cats.groupby(LEAF)[META].first().to_frame()
    labels = {k: '' if v in NO_LABEL else v for k, v in META_LABELS.items()}
    meta['label'] = meta.squeeze().apply(lambda k: labels[k])

    # w2v features
    tojoin = []
    for role in [BYR, SLR]:
        path = FEATS_DIR + 'w2v_{}.pkl'.format(role)
        tojoin.append(unpickle(path))

    w2v = pd.concat(tojoin, axis=1).reindex(index=leaf_ct.index)
    w2v[w2v.isna()] = 0.

    # reduce using t-SNE
    v = TSNE(n_jobs=-1, random_state=0).fit_transform(w2v.values)
    tsne = pd.DataFrame(v, columns=['x', 'y'], index=w2v.index)
    tsne = pd.concat([meta, leaf_ct, tsne], axis=1)
    tsne = tsne.reset_index().set_index([META, LEAF]).sort_index()

    return tsne


def main():
    # tsne coordinates
    tsne = do_tsne()

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
