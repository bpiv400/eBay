import pickle
from rlenv.interface.model_names import MODELS
from rlenv.env_utils import load_featnames
from constants import INPUT_DIR

NO_SUFFIX = {'start_price_pctile', 'auto_decline', 'auto_accept',
             't1', 't2', 't3'}


def save_featnames(full_name, featnames):
    featnames_path = '{}featnames/{}.pkl'.format(INPUT_DIR, full_name)
    pickle.dump(featnames, open(featnames_path, 'wb'))


def fix_featnames(featnames):
    for set_name, feat_set in featnames['x'].items():
        if 'offer' in set_name:
            turn_num = int(set_name[-1:])
            new_feats = list()
            for feat in feat_set:
                if feat not in NO_SUFFIX:
                    new_feats.append('{}_{}'.format(feat, turn_num))
                else:
                    new_feats.append(feat)
            print(new_feats)
            featnames['x'][set_name] = new_feats
    return featnames


def main():
    for mod in MODELS:
        featnames = load_featnames(mod)
        fix_featnames(featnames)
        save_featnames(mod, featnames)


if __name__ == '__main__':
    main()