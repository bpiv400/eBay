

    # normalized offer
    df['norm'] = offers.stack() / offers[0]
    # change from previous offer within role
    change = pd.DataFrame(index=offers.index)
    change[0] = 0
    change[1] = 0
    for i in range(2, 8):
        change[i] = abs(offers[i] - offers[i-2])
    df['change'] = change.stack().sort_index()
    # gap between successive offers
    gap = pd.DataFrame(index=offers.index)
    gap[0] = 0
    for i in range(1, 8):
        gap[i] = abs(offers[i] - offers[i-1])
    df['gap'] = gap.stack().sort_index()


    if f[j].__name__ in ['slr_min', 'byr_max']:
                feat = feat.div(L['start_price'])
                feat = feat.reorder_levels(
                    LEVELS + ['thread', 'index', 'clock'])

    

    # initialize output dataframe
    tf = subset[['clock']]
    cols = ['slr_offers', 'slr_best', 'byr_offers', 'byr_best']
    cols += [c + '_open' for c in cols]
    tf = tf.assign(**dict.fromkeys(cols, 0))
    # offer counts by role, excluding focal thread
    for role in ['slr', 'byr']:
        tfname = '_'.join([role, 'offers'])
        print('\t%s' % tfname)
        tf[tfname] = num_offers(subset, role)
    # error checking
    assert (tf.byr_offers >= tf.slr_offers).min()
    # set norm to 0 for last clock time observations in lstg
    last = subset.clock == subset.clock.groupby('lstg').transform('max')
    subset.loc[last[last].index, 'norm'] = 0.0
    # number of threads in each lstg
    N = subset.reset_index('thread').thread.groupby('lstg').max()
    # largest total concessions observed so far
    print('\t[slr_best, byr_best]')
    for n in range(1, N.max()+1):
        df = subset[['byr', 'norm']]
        if n > 2:
            df = df.loc[N[N >= n].index]
        idx = df.index[df.index.isin([n], level='thread')]
        for role in ['slr', 'byr']:
            tf.loc[idx, role + '_best'] = past_best(
                df, role, index=idx)