"""
listing.py

Defines classes related to encapsulation of listing data
for training reinforcement learner

Classes:
    ListingEnvironment: class that encapsulates functionality relating to
    converting raw chunked pandas inputs into Listing objects and storing them
    & bundling listings into batches
    Listing: class that encapsulates functionality related to querying constant
    and time valued data from listings
"""
import pickle
import os
import pandas as pd
import numpy as np
from util import unpickle, extract_datatype, is_none
# from tree import AVLTree


class ListingEnvironment:
    '''
    Notes to Etan:
    1. Assumes the time valued features are stored in chunked pickled dictionaries in
     a data/time_chunks/... directory where each file has
    the name dataset-chunkNumber_time.pkl. (e.g. train-42_time.pkl).
    Each dictionary contains 3 keys:
        timedf: dataframe containing all constant feature entries for listings in this chunk
        rlfeats: list of strings giving the names of the time features exposed
        to the reinforcement learner
        simfeats: list of strings giving the names of the time features exposed to the simulator

    2. Assumes the time features dataframe contains 2 columns 'anon_item_id'
    and 'time' and that rows are uniquely identifiably by this pair of columns,
    such that each row gives the time-valued features for a particular timestep of
    a particular listings. 'anon_item_id' should be an integer
    corresponding to the id as its given in the listing files,
    and 'time' should be of type pd.datetime64. Remaining columns should fully specify
    time valued features. Expects these values to be columns rather than indices

    3. Assumes the constant features dataframe does contains all constant features.
     Assumes these features are not normalized

    4. Assumes constant features data frame is stored in
    data/exps/data_name/consts/dataset-chunkNumber_consts.pkl.
    Assumes the constants pickle contains a dictionary with three entries:
        consts: pd.DataFrame containing all constant features for
        the reinforcement learner and simulator
        rlfeats: list of names of constant features exposed to the reinforcement learner
        simfeats: list of names of constant features exposed to simulator

    5. Tree data structure assumes no two entries for the same listing have the same datetime tag
    (Throws an error if they do)

    6. Does NOT populate data with model's byr_us and byr_hist parameters.
    We can perform this operation at initialization time, so we don't
    need to make separate datasets for different model parameters.
    The cost incurred at training time is made up for by creating fewer datasets

    TODO:
    Debug
    Finish documentation
    '''

    def __init__(self, data_name=None, chunk=None):
        '''
        Generates listing binaries for the first chunk of listings in a given dataset,
        then saves the encapsulating ListingEnvironment class to a pickle binary

        Attributes:
            batch: list of listing objects representing current minibatch
            base_dir: path to current dataset's directory
            datatype: one of train, test, or toy
            dir: dynamically computed string giving path to directory containing current
            datatype
            rl_consts: numpy array of integers giving column-wise indices of rl
            constant features
            sim_consts: numpy array of integers giving column-wise indices of
            simulator features
            rl_time: numpy array of integers giving column-wise indices of rl
            time-valued features
            sim_time: numpy array of integers giving column-wise indices of simulator
            time-valued features
            constscols: pd.Series where index gives names of constant columns
            and data gives the corresponding indices in the consts vectors
            timecols: pd.Series where index gives names of time valued feature
            columns and data gives the corresponding indices in the Listing.time
            nd.arrays

        Class Methods:
            load:

        Instance Methods:
            save:
            load_listing:
            load_batch:
            gen_data:
            query_time:
            init:
            get_sale_time:
            get_sale_price:
            get_bin_prices:
            pickle_listing:
        '''
        # ensure arguments are defined
        is_none(data_name, name='data_name')
        is_none(chunk, name='chunk')
        # initialize name tracking variables for finding the base directory
        # containing all listing subdirectories (train, test, toy)
        self.base_dir = 'data/datasets/%s/' % data_name
        # path to environment encapsulating
        env_file = '%slistings/env.pkl' % self.base_dir
        # initialize empty batch list
        self.batch = []
        self.ids = None
        # initialize name tracking variable for the datatype (train, test, toy)
        self.datatype = extract_datatype(chunk)
        # generate data for the initial chunk
        self.gen_data(chunk)
        # store environment in pickle if one doesnt exist
        self.save()

    def save(self):
        '''
        Stores environment in pickle if one doesn't already exist
        '''
        f = open('%s/env.pkl' % self.dir, 'wb')
        pickle.dump(self, f)
        f.close()

    @classmethod
    def load(cls, data_name):
        '''
        Loads environment file from pickle and throws IOError if one doesn't exist
        '''
        # define environment file
        env_file = 'data/datasets/%s/listings/env.pkl' % data_name
        # throw an error if it doesn't exist
        if not os.path.isfile(env_file):
            raise IOError(
                "%s does not exist. Have you generated the listing files?")
        else:
            f = open(env_file, 'rb')
            out = pickle.load(f)
            return out

    @property
    def dir(self):
        return self.base_dir + 'listings/' + self.datatype + '/'

    def load_listing(self, ix):
        '''
        Loads a particular listing. Throws an IOError if the target listing
        does not exist
        '''
        path = '%s/%d.pkl' % (self.dir, ix)
        if not os.path.isfile(path):
            raise IOError("Listing file does not exist in %s for %d" %
                          (self.dir, ix))
        else:
            f = open(path, 'rb')
            listing = pickle.load(f)
            f.close()
            return listing

    def load_batch(self, ids):
        '''
        Loads a batch of listings

        Args:
            ids: numpy array of item_id's (thread ids)
        '''
        self.batch = [self.load_listing(ix) for ix in ids]
        self.ids = ids

    def gen_data(self, chunk, new_env=True):
        """
        Generates listing binaries for a particular chunk
        """
        # load consts data
        consts_file = '%sconsts/%s/%s_consts.pkl' % (
            self.base_dir, self.datatype, chunk)
        consts_dict = unpickle(consts_file)
        # extract dictionary contents and delete dictionary
        constsdf = consts_dict['consts']  # shared features
        # rl specific features (string list)
        rl_consts = consts_dict['rlfeats']
        # simulator specific features (string list)
        sim_consts = consts_dict['simfeats']

        # ensure df has expected index
        if constsdf.index.name != 'item':
            raise ValueError('consts must have item as index')
        # sort columns alphabetically
        constsdf = constsdf.reindex(sorted(constsdf.columns), axis=1)
        # create map for constant feature lookup later
        constsind = pd.Series(index=constsdf.index,
                              data=np.arange(len(constsdf.index)))
        # create map for expected constant column lookup later
        constscols = pd.Series(index=constsdf.columns,
                               data=np.arange(len(constsdf.columns)))
        # if this is a new listing environment create class variable
        if new_env:
            self.constscols = constscols
        # otherwise make sure the existing class variable agrees with current chunk
        else:
            if not constscols.equals(self.constscols):
                raise ValueError(
                    "const dataframe of original chunk doesn't match current chunk")
        # sort rl_consts and sim_consts
        rl_consts = sorted(rl_consts)
        sim_consts = sorted(sim_consts)
        # convert rl_consts and sim_consts to index numbers
        rl_consts = self.constscols.loc[rl_consts].values
        sim_consts = self.constscols.loc[sim_consts].values
        # compare rl_consts & sim_consts to stored values if this is not a new environment
        if new_env:
            self.rl_consts = rl_consts
            self.sim_consts = sim_consts
        else:
            if (not np.array_equal(self.rl_consts, rl_consts) or
                    not np.array_equal(self.sim_consts, sim_consts)):
                raise ValueError(
                    "RL consts or Simulator consts doesn't equal stored value")

        constsdf = constsdf.values
        # load time features pickle
        time_dir = '%stime/%s/time_%s' % (self.base_dir, self.datatype, chunk)
        time_dict = unpickle(time_dir)
        # extract keys
        rl_time = time_dict['rlfeats']
        sim_time = time_dict['simfeats']
        timedf = time_dict['timedf']
        # check whether the index is set correctly
        if timedf.index.names is None or timedf.index.names[0] != 'item' or timedf.index.names[1] != 'clock':
            raise ValueError(
                'time dataframe index expected to be set in previous processing steps')
        # reset index
        timedf.reset_index(inplace=True, drop=False)
        # sort time valued feature columns alphabetically
        timedf = timedf.reindex(sorted(timedf.columns), axis=1)
        # create series to map time feautre to index in matrix
        timecols = pd.Series(index=timedf.columns,
                             data=np.arange(len(timedf.columns)))
        # if this is a new environment, create new class variable for time column
        if new_env:
            self.timecols = timecols
        else:
            # otherwise check that the time column map is the same as the stored one
            if not timecols.equals(self.timecols):
                raise ValueError(
                    "current time map disagrees with original map")
        # sort rl and sim time features
        rl_time = sorted(rl_time)
        sim_time = sorted(sim_time)
        # if this is a new environment, create class variables
        if new_env:
            self.rl_time = rl_time
            self.sim_time = sim_time
        else:
            # otherwise ensure agreement between current and previous sim/rl time features
            if (not np.array_equal(self.rl_time, rl_time) or
                    not np.array_equal(self.sim_time, sim_time)):
                raise ValueError(
                    "current rl or sim time feats disagree with original")

        # group into threads
        listing_groups = timedf.groupby('anon_item_id')
        # extract time from timedf
        timecol = timedf['time'].values
        timedf.drop(columns='time', inplace=True)
        # convert data frame to np.array
        timedf = timedf.values
        # create item id output list
        ids = []
        for item_id, curr_ix in listing_groups.groups.items():
            # check whether the listing is contained in the consts data frame
            try:
                constsind.loc[item_id]
            except KeyError:
                print("Consts is missing %d" % item_id)
                continue  # if not, skip the rest of this iteration
            # extract constant features from consts matrix
            curr_consts = constsdf[constsind.loc[item_id], :]
            # generate a listing for the item
            curr_listing = Listing(timefeats=timedf[curr_ix, :], constfeats=curr_consts,
                                   timecol=timecol[curr_ix],
                                   sale=curr_consts[constscols['item_price']],
                                   id=item_id)
            self.pickle_listing(curr_listing, data_name=data_name, id=item_id)
            # add current item ids to list of all item ids in dataset
            path = 'data/datasets/%s/listings.txt' % data_name
            ids.append(item_id)
        # add current ids to list of listing ids for current dataset
        if new_env:
            write_type = 'w'
        else:
            write_type = 'a'
        with open(path, write_type) as f:
            for item_id in ids:
                f.write('%d\n' % item_id)

    def pickle_listing(self, listing, **kwargs):
        """
        Pickle a particular listing binary

        Expects keywords: id, data_name
        """
        pard = 'data/datasets/%s/listings' % kwargs['data_name']
        if not os.path.isdir(pard):
            os.makedirs(pard)
        pick = open('%s/%d' % (pard, kwargs['id']), 'wb')
        pickle.dump(listing, pick)
        pick.close()

    def get_sale_time(self):
        """
        Returns the sale time in seconds after the listing was posted for each listing
        as a 1-dimensional numpy array
        """
        times = np.zeros(len(self.batch))  # null output structure
        # find starting price locations among constant features
        for i, listing in enumerate(self.batch):
            times[i] = listing.end_offset  # grab price from each
        return times

    def get_sale_price(self):
        """
        Returns the sale price in dollars for each listing as a 1-dimensional
        numpy array
        """
        prices = np.zeros(len(self.batch))  # null output structure
        # find starting price locations among constant features
        for i, listing in enumerate(self.batch):
            prices[i] = listing.sale  # grab price from each
        return prices

    def get_bin_prices(self):
        """
        Returns the buy-it-now price for each listing
        as a 1-dimensional numpy array
        """
        prices = np.zeros(len(self.batch))  # null output structure
        # find starting price locations among constant features
        ix = self.constscols.loc['start_price_usd']
        # iterate over listings
        for i, listing in enumerate(self.batch):
            prices[i] = listing.init(ix)  # grab price from each
        return prices

    def init(self, sim=False):
        """
        Initializes constant features for current batch

        Kwargs:
            sim: Boolean giving whether time valued features are being
            queried for simulator (false implies rl)
        """
        if sim:
            feats = self.sim_consts
        else:
            feats = self.rl_consts
        out = np.zeros((len(self.batch), feats.size))
        for i, listing in enumerate(self.batch):
            out[i, :] = listing.query_consts(ix=feats)
        return out

    def query_time(self, ids, delays, sim=False):
        """
        Return time valued features as 2-dimensional numpy array

        Args:
            ids: numpy array giving the indices in the batch for which
            time valued features are being queried
            delays: numpy array giving time delay (IN SECONDS) for each
            listing in ids
        Kwargs:
            sim: Boolean giving whether time valued features are being
            queried for simulator (false implies rl)
        """
        if sim:
            feats = self.sim_time
        else:
            feats = self.rl_time
        out = np.zeros((ids.size, feats.size))
        for i, ind in enumerate(ids):
            listing = self.batch[ind]
            out[i, :] = listing.query_time(delays[i], ix=feats)
        return out


class Listing:
    '''
    Encapsulates data about a particular listing, including time valued features
    and constant features for both reinforcement learner and the simulator

    Attributes:
        sale: gives sale price of the item
        end_time: last time observed for listing (pd.datetime)
        start_time: first time observed for listing (pd.datetime)
        time: 2-dimensional np.array giving, where each row gives a time-step and each column
        gives a feature
        timecol: 1 dimensional np.array giving times of each offer step, starting with the creation
        of the listing
        constfeats: 1 dimensional np.array containing constant features for the listing

    '''

    def __init__(self, **kwargs):
        '''
        Assumes the time column consists of dates, rather than second offsets from the
        moment the listing was posted

        Kwargs:
            sale: np.float64 giving the unnormalized price the item was sold for
            end_offset: number of seconds for which the listing was posted
            timefeats: np.array containing time valued features for the listing
            constfeats: 1 dimensional np.array containing constant features for the listing
            timecol: 1 dimensional np.array giving times of each offer step, starting
            with the creation of the listing
            id: np.int64 giving the listing id number in-case it's needed for something
        '''
        self.sale = kwargs['sale']
        # self.tree = AVTTree()

        # error checking for time features
        if kwargs['timefeats'] is None or kwargs['timecol'] is None:
            raise ValueError('timefeats must be provided')
        self.timecol = kwargs['timecol']
        self.time = kwargs['timefeats']
        self.consts = kwargs['constfeats']
        self.end_time = self.timecol.max()
        self.start_time = self.timecol.min()
        self.end_offset = (self.end_time - self.start_time).total_seconds()
        self.id = kwargs['id']

    def init(self, ix):
        '''
        Creates vector of features to initialize the RL or simulator
        '''
        return self.consts[ix]

    def query_time(self, offset, **kwargs):
        '''
        Returns np.ndarray of time valued features corresponding to the
        given names or indices. Expects one of names or ix keyword
        arguments to be given

        Args:
            offset: number of seconds since start of listing, given as int
            or np.int64
        Kwargs:
            ix: list or numpy array of column indices for time valued features
            names: list or numpy array of strings
        Returns:
            1 dimensional np.ndarray of time valued features
        '''
        if kwargs['ix'] is None:
            raise ValueError('names or ix must be given')

        # convert offset to seconds data type and add it to initial time
        date = self.start_time + np.timedelta64(offset, 's')
        # return none if the thread is over
        if date > self.end_time:
            return None
        # subtract current date from all dates
        query = self.timecol - date
        # subset to only dates before given date
        query = query.loc[query[query <= 0].index]
        # extract the index of the max (ie the date immediately
        # before the subtracted date)
        query = query.idxmax()
        # extract ix features from query row and return
        query = self.time[query, kwargs['ix']]
        return query
