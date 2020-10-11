import pandas as pd
from agent.util import load_values
from utils import unpickle, topickle, load_file
from agent.const import DELTA_CHOICES
from constants import FEATS_DIR, PLOT_DIR, TEST
from featnames import META, LEAF, LOOKUP


def main():
    # tsne coordinates
    tsne = unpickle(FEATS_DIR + 'tsne.pkl')

    # add meta colors
    cats = tsne.index.get_level_values(META).unique()
    colors = pd.Series(range(len(cats)), index=cats, name='c')

    # join and put in dictionary
    d = {'w2v_meta': tsne.join(colors, on=META)}

    # use values for color
    leaf = load_file(TEST, LOOKUP)[LEAF]
    for delta in DELTA_CHOICES:
        vals = load_values(part=TEST, delta=delta)
        mean_vals = vals.groupby(leaf).mean().rename('c')
        d['w2v_delta_{}'.format(delta)] = tsne.join(mean_vals, on=LEAF)

    # save
    topickle(d, PLOT_DIR + 'w2v.pkl')


if __name__ == '__main__':
    main()
