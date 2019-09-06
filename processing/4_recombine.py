"""
Recombines chunks and splits into train, dev, sim, and test.
"""

import os, random, pickle
import pandas as pd, numpy as np

DIR = 'data/chunks'
SEED = 123456
SHARES = {'train_models': 1/3, 'train_rl': 1/3}
OUT_PATH = 'data/partitions/'


def save_partitions(slr_dict, data, name):
    for key, val in slr_dict.items():
        print('Saving %s/%s' % (key, name))
        # create directory if one does not exist
        if not os.path.exists(OUT_PATH + key):
            os.mkdir(OUT_PATH + key)
        # index of lstgs to slice with
        idx = pd.Index(val, name='lstg')
        # write path
        path = OUT_PATH + '%s/%s.pkl' % (key, name)
        # reindex contents of dictionary
        out = {}
        if name == 'y':
            for m, d in data.items():
                out[m] = {k: v.reindex(
                    index=idx, level='lstg') for k, v in d.items()}
        elif name in ['x', 'z']:
            for k, v in data.items():
                if len(v.index.names) == 1:
                    out[k] = v.reindex(index=idx)
                else:
                    out[k] = v.reindex(index=idx, level='lstg')
        # write to pickle
        print(out)
        pickle.dump(out, open(path, 'wb'))


def append_chunks(paths, name):
    # initialize output
    out = {}
    # loop over chunks
    for path in sorted(paths):
        chunk = pickle.load(open(path, 'rb'))
        if name == 'y':
            for m, d in chunk[name].items():
                if m not in out:
                    out[m] = {}
                for outcome, val in d.items():
                    if outcome in out[m]:
                        out[m][outcome] = out[m][outcome].append(
                            val, verify_integrity=True)
                    else:
                        out[m][outcome] = val
        elif name in ['x', 'z']:
            for k, v in chunk[name].items():
                if k in out:
                    out[k] = out[k].append(v, verify_integrity=True)
                else:
                    out[k] = v
    return out


def randomize_lstgs(u):
    random.seed(SEED)   # set seed
    np.random.shuffle(u)
    slr_dict = {}
    last = 0
    for key, val in SHARES.items():
        curr = last + int(u.size * val)
        slr_dict[key] = sorted(u[last:curr])
        last = curr
    slr_dict['test'] = sorted(u[last:])
    return slr_dict


if __name__ == '__main__':
    # list of feats files
    paths = ['%s/%s' % (DIR, name) for name in os.listdir(DIR)
        if os.path.isfile('%s/%s' % (DIR, name)) and 'feats' in name]

    # loop over data
    for name in ['x', 'y', 'z']:
        d = append_chunks(paths, name)

        # randomize listings
        if name == 'x':
            lstgs = np.unique(d['lstg'].index.get_level_values(
                level='lstg'))
            slr_dict = randomize_lstgs(lstgs)

        # save to pickle
        save_partitions(slr_dict, d, name)

        # save some RAM
        del d
