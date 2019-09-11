"""
Class for storing and updating time valued features

TODO:
1. Update expiration queue and heap if necessary
2. Double check expiration queue and heap tests to ensure they still pass
3. Add expiring features to TimeFeatures object

"""

import math
from collections import deque, Counter
import numpy as np
import rlenv.time_triggers as time_triggers
import rlenv.env_constants as consts


class TimeFeatures:
    """
        Object for updating and setting time valued features
        Current list of features:
            - slr offers: number of seller offers in other threads, excluding rejects.
            - byr offers: number of buyer offers in other threads, excluding rejects.
            - slr offers open: number of open, unanswered slr offers in other open threads.
            - byr offers open: number of open, unanswered byr offers in other open threads.
            - slr best: the highest seller total concession in other threads, defined as 1-offer/start_price
            - byr best: the highest buyer total concession in other threads, defined as offr/start_price
            - slr best open: the highest unanswered seller concession among open seller offers.
            - byr best open: the highest unanswered buyer total concession among open buyer offers.
        Attributes:
            offers)
            feats: dictionary containing time valued features for each listing. Values are dictionaries
            containing data structures and values required to calculate all time valued features in list
        Public functions:
            get_feats: returns a dictionary of time features for a given set of ids
            lstg_active: checks whether a given lstg is still open (e.g. not sold in some other thread
            or expired)
            initialize_time_feats: initializes self.feats with all objects necessary to contain a new
            lstg
            update_features: dispatches helper update function to update all time features relevant to
            a given event
        Private functions:
            _get_feat: helper function for get_feats that returns value of a specific level-feature
            _initialize_feature: helper function for initialize_time_feats
            _update_thread_close: updates features after thread closes for some reason (expiration or byr rejection)
            _update_thread_expire: updates features after a thread has expired (time out of byr or slr)
            _update_byr_rejection: updates features after byr rejects an offer, closing thread
        """
    def __init__(self):
        """
        Initializes features object

        """
        super(TimeFeatures, self).__init__()
        self.feats = dict()

    def lstg_active(self, lstg):
        """
        Returns whether a particular listing is open, given that listing's
        id dictionary. This gives the function the flexibility to check whether
        the listing associated with some thread is open more easily

        :param lstg: integer giving unique identifier for the lstg
        :return: Bool
        """
        return lstg in self.feats

    @staticmethod
    def _get_feat(thread_id=None, feat=None, feat_dict=None, time=0):
        """
        Gets the value of a specific feature for a thread

        :param thread_id: current thread id if any
        :param feat: string giving name of the target feature
        :param feat_dict: dictionary where the feature value is stored
        :return: Numeric giving feature
        """
        args = {
            'exclude': thread_id
        }
        if 'recent' in feat:  # ignored for the timebeing
            args['time'] = time
        return feat_dict[feat].peek(**args)

    def get_feats(self, ids, time):
        """
        Gets the time features associated with a specific lstg

        :param ids: tuple giving identifiers for the event (lstg, thread_id)
        :param time: integer giving current time
        :return: array of time-valued features in the order given by env_consts.TIME_FEATURES
        """
        if ids[0] not in self.feats:
            raise RuntimeError('lstg not active')
        output = {}
        if len(ids) > 1:
            thread_id = ids[1]
        else:
            thread_id = None

        for feat in consts.TIME_FEATS:
            output[feat] = self._get_feat(thread_id=thread_id, feat=feat,
                                          feat_dict=self.feats[ids[0]],
                                          time=time)
        return output

    @staticmethod
    def _initialize_feature(feat):
        """
        Initializes a given feature
        :param feat: name of the feature
        :return: Object encapsulating feature
        """
        if 'best' not in feat:
            if 'recent' in feat:  # will be ignored for the timebeing
                return ExpirationQueue(expiration=consts.EXPIRATION)
            else:
                return FeatureCounter(feat)
        else:
            if 'recent' in feat:  # will be ignored for the timebeing
                return ExpirationHeap(min_heap=False,
                                      expiration=consts.EXPIRATION)
            else:
                return FeatureHeap(min_heap=False)

    def initialize_time_feats(self, lstg):
        """
        Creates time feats for a listing, assuming the time feats up to this point
        have been computed correctly.

        :param lstg: id of the lstg being initialized
        """
        self.feats[lstg] = dict()
        for feat in consts.TIME_FEATS:
            self.feats[lstg][feat] = TimeFeatures._initialize_feature(feat)

    # noinspection PyUnusedLocal
    @staticmethod
    def _update_thread_close(feat_dict=None, feat_name=None, offer=None, thread_id=None):
        """
        Updates time features to reflect the thread for the given event closing

        :param feat_dict: dictionary of time valued features for the current level
        :param offer: dictionary giving parameters of the offer
        :param feat_name: string giving name of time valued feature
        :param thread_id: id of the current thread
        :return: NA
        """
        if 'recent' not in feat_name:  # always true for the time being
            if 'open' in feat_name:
                feat_dict[feat_name].remove(thread_id=thread_id)

    @staticmethod
    def _update_offer(feat_dict=None, feat_name=None, offer=None, thread_id=None):
        """
        Updates value of feature after buyer or seller concession offer (not acceptance
        or rejection)

        :param feat_dict: dictionary of time valued features for the current level
        :param offer: dictionary giving parameters of the offer
        :param feat_name: string giving name of time valued feature
        :param thread_id: id of the current thread where there's been an offer
        :return: NA
        """
        if offer['type'] in feat_name:
            if 'best' not in feat_name:
                feat_dict[feat_name].increment(thread_id=thread_id)
            else:
                args = {
                    'value': offer['price'],
                    'thread_id': thread_id
                }
                if 'recent' in feat_name:  # ignored for time being
                    args['time'] = offer['time']
                feat_dict[feat_name].promote(**args)
        elif 'open' in feat_name and 'recent' not in feat_name:  # temporary do not consider recent features
            feat_dict[feat_name].remove(thread_id=thread_id)

    @staticmethod
    def _update_slr_rejection(feat_dict=None, feat_name=None, offer=None, thread_id=None):
        """
        Updates time valued features with the result of slr rejection at turn 2 or 4
        - Increments number of open slr offers
        - Does not increment number of total slr offers
        - Updates open best slr offer
        - Closes previously open buyer offer in the same thread (decrements
        number of open buyer offers and removes best open buyer offer)

        :param feat_dict: dictionary of time valued features for the current level
        :param offer: dictionary giving parameters of the offer
        :param feat_name: string giving name of time valued feature
        :param thread_id: id of the current thread where there's been an offer
        :return: NA
        """
        if offer['type'] in feat_name:
            if 'best' not in feat_name and 'open' in feat_name:
                feat_dict[feat_name].increment(thread_id=thread_id)
            # will never update slr best since it's the same as previous value
            elif 'open' in feat_name:
                args = {
                    'value': offer['price'],
                    'thread_id': thread_id
                }
                if 'recent' in feat_name:  # ignored for time being
                    args['time'] = offer['time']
                feat_dict[feat_name].promote(**args)
        elif 'open' in feat_name and 'recent' not in feat_name:
            # temporary do not consider recent features
            feat_dict[feat_name].remove(thread_id=thread_id)

    def update_features(self, trigger_type=None, ids=None, offer=None):
        """
        Updates the time valued features after some type of event

        :param ids: tuple giving identifiers for the event (lstg, thread_id)
        :param offer: dictionary containing time of offer (time),
         count of offer in the thread (count), and price (price)
        :param trigger_type: string defining event type
        :return: NA
        """
        if trigger_type == time_triggers.OFFER:
            feature_function = TimeFeatures._update_offer
        elif trigger_type == time_triggers.BYR_REJECTION:
            feature_function = TimeFeatures._update_thread_close
        elif trigger_type == time_triggers.SLR_REJECTION:
            feature_function = TimeFeatures._update_slr_rejection
        elif trigger_type == time_triggers.LSTG_EXPIRATION:
            del self.feats[ids[0]]  # just delete the lstg
            return None
        elif trigger_type == time_triggers.SALE:
            [print(key) for key in self.feats]
            del self.feats[ids[0]]  # just delete the lstg
            [print(key) for key in self.feats]
            return None
        else:
            raise NotImplementedError()
        for feat in consts.TIME_FEATS:
            feature_function(feat_dict=self.feats[ids[0]], feat_name=feat,
                             offer=offer, thread_id=ids[1])

    def __contains__(self, item):
        """
        Checks whether a lstg is active right now
        :param item: lstg id
        :return: NA
        """
        return self.lstg_active(item)

    def __delitem__(self, key):
        """
        Delete lstg from time feats object
        :param key: lstg id
        :return: NA
        """
        del self.feats[key]



# noinspection DuplicatedCode
class FeatureHeap:
    """
    Attributes:
        index_map: dictionary mapping thread_id's to indexes of corresponding elements in heap
        heap: array of HeapEntry's representing the heap
        min: boolean denoting whether this is a min heap (max if not)
        size: integer giving the number of elements in the heap
    Private Functions:
        _swap: swaps elements at the given indices in the heap
        _get_true_parent: returns which of three elements (a parent and its children) should be the true parent
        _bad_order: returns whether two elements (a parent and child) are out of order
    Public functions:
        pop: pops the top element off the heap
        peek: peeks at the top element on the heap (excluding a given entry)
        demote: updates the value of an entry, pushing it lower in the heap
        promote: updates the value of an entry, pushing it higher in the heap
        remove: removes a given entry from the heap
        push: adds a given entry to the heap
    """

    def __init__(self, min_heap=False):
        """
        Constructor

        :param min_heap: boolean indicating whether this should be a min heap (max if not)
        """
        super(FeatureHeap, self).__init__()
        self.heap = []
        self.size = 0
        self.index_map = dict()
        self.min = min_heap

    def peek(self, exclude=None):
        """
        peeks at the value on the top of the heap, excludes this value if has the id of the excluded thread

        :param exclude: thread_id of an entry that should not be considered among candidate elements for the min(max)
        :return: value of the min(max) element. If the heap is empty, a max heap returns 0 and a min heap returns -1
        """
        # error handling
        if self.size == 0:
            if self.min:
                return -1
            else:
                return 0
        if exclude not in self:
            exclude = None
        if exclude is None or self.index_map[exclude] != 0:
            return self.heap[0].value
        else:
            if self.size >= 3 and self.min:
                return min(self.heap[1].value, self.heap[2].value)
            elif self.size >= 3 and not self.min:
                return max(self.heap[1].value, self.heap[2].value)
            elif self.size == 2:
                return self.heap[1].value
            else:
                if not self.min:
                    return 0
                else:
                    return -1

    # noinspection DuplicatedCode
    def promote(self, thread_id=None, value=None):
        """
        Updates the value of the entry with the given thread_id and maintains heap order
        
        :param value: updated value of thread_id. None implies
        value has already been updated and only the swapping must occur
        :param thread_id: integer giving the unique id of the target entry
        :return: Boolean for whether the value updated (false implies the promotion was actually a demotion)
        True implies value is the same as current value or more extreme (i.e. closer to the top element)
        """
        if thread_id not in self:
            self.push(thread_id=thread_id, value=value)
        else:
            index = self.index_map[thread_id]
            # determine cases when promotion is actually required
            if self.min:
                if self.heap[index].value > value:
                    self.heap[index].value = value
                else:
                    return not self.heap[index].value < value
            else:
                if self.heap[index].value < value:
                    self.heap[index].value = value
                else:
                    return not self.heap[index].value > value
            child_index = index
            while True:
                parent_index = math.ceil(child_index / 2) - 1
                if self._bad_order(child_index, parent_index) and child_index != 0:
                    self._swap(child_index, parent_index)
                    child_index = parent_index
                else:
                    return True

    def pop(self):
        """
        Pops the top element off the heap and ensures heap order is maintained

        :return: HeapEntry of the top element
        :raises: RuntimeError if the heap is empty
        """
        if self.size == 0:
            raise RuntimeError("Removing from empty heap")

        idx = self.heap[0].thread_id
        return self.remove(thread_id=idx)

    # noinspection DuplicatedCode
    def remove(self, thread_id=None):
        """
        Removes entry associated with given thread_id from the heap

        :param thread_id: integer giving the unique id of the target entry
        :return: HeapEntry for the removed item
        """
        if thread_id not in self.index_map:
            return None
        if self.size == 1:
            self.size = 0
            self.index_map = dict()
            last_item = self.heap[0]
            self.heap = []
            return last_item
        else:
            self.size += -1
            index = self.index_map[thread_id]
            remove_item = self.heap[index]

            if index == self.size:
                self.index_map.pop(thread_id)
                del self.heap[-1]
                return remove_item
            last_item = self.heap[self.size]
            self.heap[index] = last_item
            self.heap[self.size] = remove_item
            self.index_map.pop(thread_id)
            self.index_map[last_item.thread_id] = index
            del self.heap[-1]
            self.demote(thread_id=last_item.thread_id, value=None)
            return remove_item

    def demote(self, value=None, thread_id=None):
        """
        Demotes the element with the given thread_id in the heap

        :param value: updated value of thread_id. None implies
        value has already been updated and only the swapping must occur
        :param thread_id: integer giving the unique id of the target entry
        :return: Boolean for whether the value updated (false implies remained the same)
        :raises: RuntimeError if demotion is actually a promotion
        :raises: RuntimeError if thread_id is not in the heap
        """
        if thread_id not in self:
            raise RuntimeError("May not demote non-existent thread")
        if value is not None:
            return_bool = (self.heap[self.index_map[thread_id]].value > value and not self.min) or \
                          (self.heap[self.index_map[thread_id]].value < value and self.min)
            if not return_bool:
                if self.heap[self.index_map[thread_id]].value != value:
                    raise RuntimeError("Cannot use demote to promote from %d to %d" %
                                       (self.heap[self.index_map[thread_id]].value, value))
            self.heap[self.index_map[thread_id]].value = value
        else:
            return_bool = True

        parent_index = self.index_map[thread_id]
        while True:
            first_child = (parent_index * 2) + 1
            sec_child = (parent_index * 2) + 2
            if sec_child < self.size:
                true_parent = self._get_true_parent(parent_index, first_child, sec_child)
                # three bad order
                if true_parent == 0:
                    break
                elif true_parent == 1:
                    self._swap(first_child, parent_index)
                    parent_index = first_child
                else:
                    self._swap(sec_child, parent_index)
                    parent_index = sec_child
            elif first_child < self.size:
                # two bad order
                true_parent = self._get_true_parent(parent_index, first_child, None)
                if true_parent == 1:
                    self._swap(first_child, parent_index)
                break
            else:
                break
        return return_bool

    def push(self, thread_id=None, value=None):
        """
        Pushes a value associated with a given thread onto the heap

        :param value: updated value of thread_id. None implies
        value has already been updated and only the swapping must occur
        :param thread_id: integer giving the unique id of the target entry
        :return: NA
        :raises RuntimeError: when an entry with thread_id already exists in the heap
        :raises RuntimeError: when thread_id or value is None
        """
        if thread_id is None or value is None:
            raise RuntimeError('Must specify thread_id and value')
        if thread_id in self.index_map:
            raise RuntimeError("Cannot insert thread into heap where it already exists")
        item = HeapEntry(thread_id=thread_id, value=value)
        self.heap.append(item)
        self.index_map[thread_id] = self.size
        if self.size != 0:
            child = self.size
            parent = math.ceil(child / 2) - 1
            while child != 0 and self._bad_order(child, parent):
                self._swap(child, parent)
                child = parent
                parent = math.ceil(child / 2) - 1
        self.size += 1

    def __contains__(self, item):
        """
        Returns whether a given thread_id is contained in the heap
        :param item: thread_id (expects integer)
        :return: Boolean
        """
        return item in self.index_map

    def _swap(self, child_index, parent_index):
        """
        Swaps the element stored at index child with the element stored at index parent
        Updates the index_map appropriately

        :param child_index: integer index of child element
        :param parent_index: integer index of the parent element
        :return:
        """
        child = self.heap[child_index]
        parent = self.heap[parent_index]
        self.index_map[child.thread_id] = parent_index
        self.index_map[parent.thread_id] = child_index
        self.heap[child_index] = parent
        self.heap[parent_index] = child

    def _get_true_parent(self, parent, first_child, second_child):
        """
        Returns an integer representing which of 3 items should be made the parent among them
        0 indicates parent should be parent among them, 1 indicates first child should be made
        parent, 2 indicates second child should be made parent
        :param parent: index of parent element
        :param first_child: index of first child
        :param second_child: index of second child
        :return: integer (0, 1, 2)
        """
        if second_child is not None:
            if self.min:
                return np.argmin([self.heap[parent].value,
                                  self.heap[first_child].value,
                                  self.heap[second_child].value])
            else:
                return np.argmax([self.heap[parent].value,
                                  self.heap[first_child].value,
                                  self.heap[second_child].value])
        else:
            if self.min:
                return np.argmin([self.heap[parent].value,
                                  self.heap[first_child].value])
            else:
                return np.argmax([self.heap[parent].value,
                                  self.heap[first_child].value])

    def _bad_order(self, child, parent):
        """
        Determines whether a child and it's parent are out of order, given the type of the heap
        :param child: index of child
        :param parent: index of parent
        :return: boolean indicating that the child and the parent should be swapped to maintain heap order
        """
        child = self.heap[child]
        parent = self.heap[parent]
        if self.min:
            return child < parent
        else:
            return child > parent


class HeapEntry:
    """
    Class for encapsulating object in FeatureHeap

    Attributes:
        thread_id: gives the unique identifier of the thread associated with this value
        value: value of the feature
    """

    def __init__(self, thread_id=None, value=None):
        """
        :param value: updated value of thread_id. None implies
        value has already been updated and only the swapping must occur
        :param thread_id: integer giving the unique id of the target entry
        """
        self.thread_id = thread_id
        self.value = value

    def __lt__(self, other):
        """
        Returns whether self is less than other (by value)
        
        :param other: HeapEntry 
        :return: Boolean
        """
        return self.value < other.value

    def __gt__(self, other):
        """
        Returns whether self is greater than other
        
        :param other: HeapEntry 
        :return: Boolean
        """
        return self.value > other.value

    def __repr__(self):
        """
        String representation of the HeapEntry
        :return: NA
        """
        return '(thread_id: %d, value: %d)' % (self.thread_id, self.value)


# noinspection DuplicatedCode,DuplicatedCode,DuplicatedCode
class ExpirationHeap:
    """
    Attributes:
        time_heap: min FeatureHeap where values denote the time each element was last updated 
        feat_heap: FeatureHeap where values denote the value with each element is added to the ExpirationHeap
        size: number of elements in the expiration heap
        expiration: number of units of time until elements should be removed from the heap
        backups: dictionary mapping thread_id to dequeues of (time, value) tuples where, for each thread_id, each
        tuple has a less extreme value, but more recent time stamp than than the element currently in feature_heap
        for that thread_id. The oldest tuples are the leftmost ones (append right) FIFO-style
        min: boolean giving whether this is a min expiration heap
    Public Functions:
        pop: removes the most extreme element from heap
        push: adds a new thread to expiration heap
        promote: updates an existing entry in the heap (pushes up in the heap or adds backup)
        remove: removes an existing entry from the heap
    Private functions:
        _add_backup: adds a new backup
    """

    def __init__(self, min_heap=False, expiration=None):
        """
        
        :param min_heap: Boolean giving whether this is a min heap (max if not) 
        :param expiration: numeric giving number of units of time until elements should be removed from heap
        """
        self.time_heap = FeatureHeap(min_heap=True)
        self.feat_heap = FeatureHeap(min_heap=min_heap)
        self.min = min_heap
        self.expiration = expiration
        self.backups = dict()
        self.size = 0

    # noinspection DuplicatedCode
    def pop(self):
        """
        Pops the top element off the heap and replaces it with a backup if one exists

        :return: HeapEntry of the top element
        :raises: RuntimeError if no elements are in the heap
        """
        if self.size == 0:
            raise RuntimeError('may not pop from empty heap')
        popped = self.feat_heap.pop()
        if popped.thread_id in self.backups:
            backup = self.backups[popped.thread_id].popleft()
            self.time_heap.demote(thread_id=popped.thread_id, value=backup[0])
            self.feat_heap.push(thread_id=popped.thread_id, value=backup[1])
            if len(self.backups[popped.thread_id]) == 0:
                self.backups.pop(popped.thread_id)
        else:
            self.size -= 1
        return popped

    def peek(self, time=None, exclude=None):
        """
        peeks at the element on top of the heap, and returns the value of the top element after removing all expired
        elements. Before removing a given element, checks whether any backups exist for that element and updates
        accordingly. 
        
        :param time: integer time stamp of peek 
        :param exclude: thread_id of an entry that should be ignored
        :return: value of min(max) among non-expired entries
        """
        if self.size == 0:
            return self.feat_heap.peek()
        else:
            # noinspection DuplicatedCode
            while self.size > 0 and (time - self.time_heap.peek() > self.expiration):
                thread_id = self.time_heap.heap[0].thread_id
                if thread_id not in self.backups:
                    self.time_heap.remove(thread_id=thread_id)
                    self.feat_heap.remove(thread_id=thread_id)
                    self.size -= 1
                else:
                    # get backup
                    backups = self.backups[thread_id]
                    if self.min:
                        index, backup = min(enumerate(backups), key=lambda x: x[1][1])
                    else:
                        index, backup = max(enumerate(backups), key=lambda x: x[1][1])
                    if index == len(backups) - 1:
                        del self.backups[thread_id]
                    else:
                        self.backups[thread_id] = [backups[i] for i in range(index + 1, len(backups))]
                    self.time_heap.demote(value=backup[0], thread_id=thread_id)
                    self.feat_heap.demote(value=backup[1], thread_id=thread_id)
            # default return value for empty feature
            return self.feat_heap.peek(exclude=exclude)

    def push(self, time=None, thread_id=None, value=None):
        """
        Add value for the given thread at the given time to the heap

        :param time: integer giving time associated with promotion
        :param thread_id: integer giving id of relevant thread
        :param value: new feature value
        :raises RuntimeError: when thread_id is already in the heap
        :return: NA
        """
        if thread_id in self:
            raise RuntimeError("May not push item already in heap")
        self.time_heap.push(thread_id=thread_id, value=time)
        self.feat_heap.push(thread_id=thread_id, value=value)
        self.size += 1

    def remove(self, thread_id=None):
        """
        Removes the item with the given thread_id from the heap

        :param thread_id: integer giving id of relevant thread
        :raises: RuntimeError if thread_id is not in the heap
        """
        if thread_id in self:
            self.feat_heap.remove(thread_id=thread_id)
            self.time_heap.remove(thread_id=thread_id)
            self.backups.pop(thread_id)
        else:
            raise RuntimeError('may not remove non-existent entry')

    def promote(self, time=None, thread_id=None, value=None):
        """
        Promotes a value in the feature heap & if the value
        updates, then update associated time. Otherwise, add a backup with the new time

        :param time: integer giving time associated with promotion
        :param thread_id: integer giving id of relevant thread
        :param value: new feature value
        :return: Boolean giving whether promotion occurred (false implies backup is added)
        :raises: RuntimeError if no entry with thread_id exists
        :raises: RuntimeError if time is earlier than the current time
        """
        if self.feat_heap.promote(thread_id=thread_id, value=value):
            self.time_heap.demote(thread_id=thread_id, value=time)
            if thread_id in self.backups:
                self.backups.pop(thread_id)
            return True
        else:
            self._add_backup(thread_id=thread_id, value=value, time=time)
            return False

    def _add_backup(self, time=None, thread_id=None, value=None):
        """
        Helper function that adds a backup entry for the given thread_id with the given time, value pair

        :param time: timestamp of current promotion
        :param thread_id: thread_id of some entry in the heap
        :param value: value associated with the new backup
        :return: NA
        """
        if thread_id not in self.backups:
            self.backups[thread_id] = deque()
        self.backups[thread_id].append((time, value))

    def __contains__(self, item):
        """
        Returns whether a given thread_id is in the heap
        :param item: integer giving thread_id
        :return: Boolean
        """
        return item in self.feat_heap


class FeatureCounter:
    """
    Class for features that require continually updating a total and excluding
    the current thread's count from that total

    Attributes:
        total: integer giving the current count
        name: name of the feature this object counts
        contents: dictionary mapping thread ids to their contribution to the
        total
    Public Functions:
        increment: Increments the counter in response to an event in some given thread
        remove: removes the contribution of a given thread from the total
        peek: gets the total excluding the current thread
    """
    def __init__(self, name):
        """
        Initializes attributes
        """
        self.total = 0
        self.name = name
        self.contents = Counter()

    def increment(self, thread_id=None):
        """
        Increments the total and the thread-specific
        counter for a given thread

        :param thread_id: id of current thread
        :return: NA
        """
        if thread_id is not None:
            self.contents[thread_id] += 1
            self.total += 1

    def remove(self, thread_id=None):
        """
        Remove contribution of some thread from the total

        :param thread_id: id of the current thread
        :return: NA
        """
        if thread_id is not None and thread_id in self.contents:
            self.total -= self.contents[thread_id]
            del self.contents[thread_id]

    def peek(self, exclude=None):
        """
        Gets the total excluding the given thread
        :param exclude: id of the current thread
        :return: integer
        """
        if exclude is None:
            return self.total
        else:
            return self.total - self.contents[exclude]


class SmallHeap:
    """
    Max heap that only contains two items for tracking historical max
    with the option of excluding 1 entry from max calculation

    Attributes:
        contents: array of len <= 2 containing the current max and second
        largest elements
    Public Functions:
        peek: gives the max excluding the current thread
        promote: updates the value associated some thread
    """
    def __init__(self):
        """
        Initialize attributes
        """
        self.contents = []

    def promote(self, thread_id=None, value=None):
        """
        Promotes the value associated with some given thread
        :param thread_id: id of thread
        :param value: new value [0, 1]
        :return: NA
        """
        ind = self._index_of(thread_id=thread_id)
        if ind == -1:
            if len(self.contents) < 2:
                self.contents.append(HeapEntry(value=value, thread_id=thread_id))
            else:
                min_index = self._min_index()
                if value > self.contents[min_index].value:
                    self.contents[min_index] = HeapEntry(value=value, thread_id=thread_id)
        else:
            if value > self.contents[ind].value:
                self.contents[ind].value = value

    def peek(self, exclude=None):
        """
        Returns the maximum value excluding the current thread id

        :param exclude: thread id to exclude
        :return: Numeric [0, 1]
        """
        max_val = 0
        for entry in self.contents:
            if entry.value > max_val and entry.thread_id != exclude:
                max_val = entry.value
        return max_val

    def _min_index(self):
        """
        Returns the index of the min
        :return: numeric
        """
        min_val = 2
        min_index = -1
        for index, entry in enumerate(self.contents):
            if entry.value < min_val:
                min_val = entry.value
                min_index = index
        return min_index

    def _index_of(self, thread_id=None):
        """
        Returns the index of the element associated with the
        given thread_id in self.contents

        :param thread_id: id of the thread to search for
        :return: NA
        """
        if thread_id is None:
            return -1
        for index, entry in enumerate(self.contents):
            if entry.thread_id == thread_id:
                return index
        return -1

    def __contains__(self, item):
        """
        Checks whether the heap contains an entry associated with the
        given thread_id
        :param item: given thread id
        :return:
        """
        for entry in self.contents:
            if entry.thread_id == item:
                return True
        return False


# noinspection DuplicatedCode
class ExpirationQueue:
    """
    Cannot handle arbitrary removal by thread_id (e.g. can't handle listing
    closings). Should only be used for time-valued expiration based features

    Attributes:
        size: number of elements in the queue at the time of last peek
        queue: deque of tuples where recent items are pushed to the right
        and expired elements are popped from the left. Each tuple gives (time, thread_id) of the entry
        expiration: number of seconds after which an item expires
        thread_counter: counter that gives the number of entries in deque for given thread_ids
    Public Functions:
        peek: counts how many elements the queue contains at a given time step, excludes count of entries
        from some given thread_id
        push: adds another element to the queue
    """

    def __init__(self, expiration=None):
        self.queue = deque()
        self.thread_counter = Counter()
        self.expiration = expiration
        self.size = 0

    def push(self, time=None, thread_id=None):
        """
        adds an element with the given time to the end of the queue

        :param time: integer denoting time of event
        :param thread_id: integer denoting the thread_id of the thread
        whose offer is being added
        :return: NA
        """
        self.queue.append((time, thread_id))
        self.thread_counter[thread_id] += 1
        self.size += 1

    def peek(self, time=None, exclude=None):
        """
        Returns the number of non-expired elements in the queue at the given time
        :param time: integer giving the current time
        :param exclude:  integer giving the thread_id whose entries should be excluded
        :return: integer giving number of elements in deque
        """
        while self.size > 0 and self.queue[0][0] + self.expiration < time:
            popped = self.queue.popleft()
            self.size -= 1
            self.thread_counter[popped[1]] -= 1
        return self.size - self.thread_counter[exclude]

    def __contains__(self, item):
        """
        Containment method always returns false so next offer will always be added

        :param item: presumably thread_id
        :return: false always
        """
        return False
