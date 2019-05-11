import numpy as np
import pandas as pd


def get_cont_index(lstgs):
    """
    Replace unique cont-item pairs with unique const index
    """
    mapping = lstgs.reset_index(drop=False)
    mapping = mapping.loc[:, ['item', 'cont']]
    mapping['def'] = True
    mapping.drop_duplicates(inplace=True, keep='first')
    mapping.reset_index(inplace=True, drop=True)
    mapping.reset_index(inplace=True, drop=False)
    mapping.rename(columns={'index': 'true_cont'}, inplace=True)
    mapping.drop(columns='def', inplace=True)
    lstgs.reset_index(inplace=True, drop=False)
    lstgs = pd.merge(lstgs, mapping, how='inner', on=['lstg', 'cont'])
    lstgs.drop(columns='cont', inplace=True)
    lstgs.rename(columns={'true_cont': 'cont'}, inplace=True)
    lstgs.drop(columns=['item', 'start', 'end', 'accept'], inplace=True)
    lstgs.set_index('lstg', inplace=True)
    return lstgs


def get_lstg_index(offers):
    curr_idx = list(offers.index.names)
    offers.reset_index(inplace=True, drop=False)
    offers.rename(columns={'lstg': 'old_lstg'}, inplace=True)
    mapping = offers.loc[:, ['slr', 'old_lstg']]
    mapping.drop_duplicates(inplace=True, keep='first')
    mapping.reset_index(inplace=True, drop=False)
    mapping.rename(columns={'index': 'lstg'}, inplace=True)
    offers = pd.merge(offers, mapping, how='inner', left_on=[
                      'slr', 'old_lstg'], right_on=['slr', 'old_lstg'])
    offers.set_index(curr_idx, inplace=True)
    return offers


def broadcast_items(targdf, sourcedf, tcol, scol):
	'''
    Broadcasts values of sourcedf through items in targdf
    '''
	sourcedf = sourcedf.reindex(index=targdf.index)
	invalids = sourcedf[sourcedf[scol].isna()].index
	sourcedf.drop(index=invalids, inplace=True)
	targdf.loc[sourcedf.index, tcol] = sourcedf[scol]
	return targdf


def get_candidate_lstgs(unsold):
    """
    Returns rows of dataframe corresponding to the listing in each item,
    for which the finish time of the current listing occurs before
    the start time of the earliest added to each listing sequence
    """
	cands = unsold.copy()
	cands['util'] = cands['end_time'] < cands['early_start']
    return cands[cands.groupby('product').cumsum() == 1]


def longest(lstgs, unsold, counter, shadow=None):
    """
    Finds the longest non-overlapping sequence of intervals
    among listings in unsold dataframe. Counter denotes the
    number of 'cont' in lstgs corresponding to the current
    sequences being built for each item

    Shadow is not null if unsold is a copy of the actual unsold
    dataframe which must be subset (i.e. in the case where we have
    subset the dataframe to only listings that began before the selling
    of the current listing occurred)
    """
    cands = get_candidate_lstgs(unsold)
    while len(cands) > 0:
        # add selected listings to the current sequences being built
        lstgs.loc[cands.index, 'cont'] = counter
        # remove from unsold and unsold subset
        unsold.drop(index=cands.index, inplace=True)
        if shadow is not None:
            shadow.drop(index=cands.index, inplace=True)
        # update value of early_start for each item
        cands.index = cands.index.droplevel('lstg')
        broadcast_items(unsold, cands, 'early_start', 'start')
        # update candidates using new list of unsold listings
        cands = get_candidate_lstgs(unsold)
    return lstgs, unsold


def extract_sold_unsold(lstgs, issold):
	df = lstgs[lstgs['accept'] == int(issold)].drop('accept', axis=1)
	df.sort_values(['slr', 'end_time'],
		ascending=[True, issold], inplace=True)
	df.reset_index(['start_time', 'end_time'],
		drop=False, inplace=True)
	return df


def add_product_index(lstgs):
	'''
    Replace slr-title-condtn with product identifer in lstgs
    (inplace) and return a mapping dataframe with columns
    slr, meta, leaf, title, cndtn, product
    '''
	mapping = lstgs.reset_index(drop=False).copy()
	mapping = mapping[['slr', 'meta', 'leaf', 'title', 'cndtn']]
	mapping.drop_duplicates(inplace=True, keep='first')
	mapping['product'] = mapping.groupby('slr').cumcount() + 1
	mapping.set_index(['slr', 'meta', 'leaf', 'title', 'cndtn'],
		inplace=True)
	lstgs = lstgs.join(mapping).set_index(
		['product', 'start_time', 'end_time'], append=True)
	lstgs.reset_index(['meta', 'leaf', 'title', 'cndtn'],
		drop=True, inplace=True)
	return lstgs.reorder_levels([0, 2, 1, 3, 4])


def add_item_index(listings):
    """
    Adds an index variable for items
    In the dataset, we have items (e.g. slr-condtn-title tuples)
    which correspond to the same item. If these appear simultaneously,
    i.e. if the listing with the earlier finish time finishes after
    the other starts
    """
	# lstgs index to ['slr', 'product', 'lstg']
	lstgs = listings.copy().drop('start_price', axis=1)
	lstgs = add_product_index(lstgs)
	lstgs['cont'] = 0
	# extract sold and unsold listings
	sold = extract_sold_unsold(lstgs, True)
	unsold = extract_sold_unsold(lstgs, False)
	lstgs.reset_index(['start_time', 'end_time'],
		drop=False, inplace=True)
	# add utility column to sold
	sold['util'] = True
	# initialize sequence counter
	counter = 1
    while len(sold) > 0:
        # indicate that the most recent sold copy for each item never sold
		unsold['sold_finish'] = np.inf
		unsold['early_start'] = np.inf
		# extract indices of sold lstgs with earliest finish time for each item
		firsts = sold[sold['util'].groupby(['slr', 'product']).cumsum() == 1]
		# add to sequence counter for each lstg
		lstgs.loc[firsts.index, 'cont'] = counter
		# remove from sold
		sold.drop(index=firsts.index, inplace=True)
        if len(unsold) > 0:
            # drop lstg from extracted index
			firsts.index = firsts.index.droplevel('lstg')
			firsts = firsts.rename(columns={
				'end_time': 'sold_finish',
				'start_time': 'early_start'})
			unsold.update(firsts)
            # subset to lstgs that started before each in-item sale
            starts = unsold[unsold['sold_finish'] > unsold['start_time']]
            longest(lstgs, starts, counter, shadow=unsold)
        counter += 1  # increment sequence counter
    # after having removed all sold listings, if any unsold listings remain,
    # greedily combine these into non-overlapping sequences of maximum length within
    # each item
    while len(unsold) > 0:
        unsold['early_start'] = np.inf
        longest(lstgs, starts, counter)
        counter += 1
    # generate a unique index for each cont-item pair in lstg and replace
    # cont with this new value
    lstgs = get_cont_index(lstgs)
    # ensure all listings are members of some continous listing
    assert (lstgs['cont'] != 0).all()
    offers = pd.merge(offers, lstgs, how='inner',
                      left_on='lstg', left_index=False, right_on='lstg')
    return offers