from deprecated.env_constants import LSTG_IDS, LSTG_FEATS_MAP


def get_ids(slr, data):
    """
    Returns a dictionary giving all of the identification categories for a given lstg
    (slr, meta, leaf, cndtn, title, lstg)

    :param slr: int giving slr id
    :param data: row of 3_environment output
    :return: Dict
    """
    out = {}
    for idx in LSTG_IDS:
        if idx == 'slr':
            out['slr'] = int(slr)
        else:
            out[idx] = int(data[LSTG_FEATS_MAP[idx]])
    out['byr_count'] = 0
    out['slr_count'] = 0
    return out
