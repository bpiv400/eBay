import argparse
import numpy as np
import pandas as pd
from compress_pickle import dump
from gensim.models import Word2Vec
from constants import BYR, SLR, SEED, CLEAN_DIR, FEATS_DIR

VOCAB_SIZE = 32  # vocabulary size for embeddings


def run_model(s):
    # construct sentences
    print('Constructing sentences')
    sentences, max_length = [], 0
    for idx in np.unique(s.index.values):
        sentence = s.loc[idx].values.tolist()
        sentences.append(sentence)
        max_length = np.maximum(max_length, len(sentence))
    # word2vec
    print('Training model')
    model = Word2Vec(sentences,
                     seed=SEED,
                     sg=1,
                     window=max_length,
                     min_count=1,
                     size=VOCAB_SIZE,
                     workers=0)
    # output dataframe
    print('Creating output')
    leafs = model.wv.vocab.keys()
    output = pd.DataFrame(index=pd.Index([], name='leaf'),
                          columns=[str(i) for i in range(VOCAB_SIZE)])
    for leaf in leafs:
        output.loc[int(leaf)] = model.wv.get_vector(leaf)
    return output.sort_index()


def main():
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--slr', action='store_true')
    slr = parser.parse_args().slr
    role = SLR if slr else BYR

    # load sentences
    filename = CLEAN_DIR + 'leaf_{}.csv'.format(role)
    s = pd.read_csv(filename).set_index(role).squeeze().astype(str)

    # run model
    df = run_model(s).rename(lambda x: role + x, axis=1).sort_index()

    # save
    dump(df, FEATS_DIR + 'w2v_{}.pkl'.format(role))


if __name__ == '__main__':
    main()
