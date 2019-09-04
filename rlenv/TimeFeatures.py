"""
Class for storing and updating time valued features

TODO:
1. Fix everything and document
2. Update testing to reflect that expiration queue now counts how many come from each thread
5. Find out from Etan whether he has signed off on the final list of time-valued features

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
            - slr offers: number of seller oers in other threads, excluding rejects.
            - byr offers: number of buyer oers in other threads, excluding rejects.
            - slr offers open: number of current open slr offers in other threads.
            - byr offers open: number of current open byr offers in other threads.
            - slr best: the highest seller total concession in other threads, defined as 1-offer/start_price
            - byr best: the highest buyer total concession in other threads, defined as offr/start_price
            - slr best open: the highest seller total concession among open seller offers.
            - byr best open: the highest buyer total concession among open buyer offers.
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
            update_features: dispatches helper update function to update all time features reelvant to
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

    def lstg_active(self, lstg_id):
        """
        Returns whether a particular listing is open
        :param lstg_id: id for a lstg
        :return: Bool
        """
        return lstg_id in self.feats

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
        if 'recent' in feat: # ignored for the timebeing
            args['time'] = time
        return feat_dict[feat].peek(**args)

    def get_feats(self, ids, time):
        """
        Gets the time features associated with a specific lstg

        :param ids: ids dictionary for a specific listing (possibly thread)
        :param time: integer giving current time
        :return: array of time-valued features in the order given by env_consts.TIME_FEATURES
        """
        output = []
        if 'thread_id' in ids:
            thread_id = ids['thread_id']
        else:
            thread_id = None

        for feat in consts.TIME_FEATS:
            output.append(self._get_feat(thread_id=thread_id,
                                         feat=feat,
                                         feat_dict=self.feats[ids['lstg_id']],
                                         time=time))
        return output

    @staticmethod
    def _initialize_feature(self, feat):
        """
        Initializes a given feature
        :param feat: name of the feature
        :return: Object encapsulating feature
        """
        if 'best' not in feat:
            if 'recent' in feat: # will be ignored for the timebeing
                return ExpirationQueue(expiration=consts.EXPIRATION)
            else:
                return FeatureCounter(feat)
        else:
            if 'recent' in feat: # will be ignored for the timebeing
                return ExpirationHeap(min_heap=False,
                                      expiration=consts.EXPIRATION)
            else:
                return FeatureHeap(min_heap=False)

    def initialize_time_feats(self, lstg_id):
        """
        Creates time feats for a listing, assuming the time feats up to this point
        have been computed correctly.

        :param lstg_id: id of the lstg being initialized
        """
        self.feats[lstg_id] = dict()
        for feat in consts.TIME_FEATS:
            self.feats[lstg_id] = self._initialize_feature(feat)

    @staticmethod
    def _update_thread_expire(feat_dict=None, feat_name=None, offer=None, thread_id=None):
        """
        Updates time valued features with result of thread expiring,
        currently just updates features to close the current thread
        (May need to be changed)

        :param feat_dict: dictionary of time valued features for the current level
        :param offer: dictionary giving parameters of the offer
        :param feat_name: string giving name of time valued feature
        :param ids: dictionary giving the ids of the current thread
        :return: NA
        """
        TimeFeatures._update_thread_close(feat_dict=feat_dict, feat_name=feat_name,
                                          offer=offer, thread_id=thread_id)

    @staticmethod
    def _update_thread_close(feat_dict=None, feat_name=None, offer=None, thread_id=None)
        """
        Updates time features to reflect the thread for the given event closing

        :param feat_dict: dictionary of time valued features for the current level
        :param offer: dictionary giving parameters of the offer
        :param feat_name: string giving name of time valued feature
        :param thread_id: id of the current thread
        :return: NA
        """
        if 'recent' not in feat_name: # always true for the time being
            if 'open' in feat_name:
                feat_dict[feat_name].remove(thread_id=thread_id)

    @staticmethod
    def _update_byr_rejection(feat_dict=None, feat_name=None, offer=None, thread_id=None):
        """
        Updates time valued features with the result of byr rejection
        (currently identical to closing a thread for any reason)

        :param feat_dict: dictionary of features containing given feat
        :param feat_name: str name of feat
        :param offer: dictionary containing details of the offer
        :param thread_id: id of the current thread
        :return: NA
        """
        TimeFeatures._update_thread_close(feat_dict=feat_dict, feat_name=feat_name,
                                          offer=offer, thread_id=thread_id)

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
                    'value' : offer['price'],
                    'thread_id' : thread_id
                }
                if 'recent' in feat_name: # ignored for time being
                    args['time'] = offer['time']
                feat_dict[feat_name].promote(**args)

    def update_features(self, trigger_type=None, ids=None, offer=None):
        """
        Updates the time valued features after some type of event

        :param ids: dictionary giving identifiers for the event
        (expects slr, meta, leaf, title, cndtn, lstg, thread_id)
        :param offer: dictionary containing time of offer (time),
         count of offer in the thread (count), and price (price)
        :param trigger_type: string defining event type
        :return: NA
        """
        feature_function = None
        if trigger_type == time_triggers.OFFER:
            feature_function = TimeFeatures._update_offer
        elif trigger_type == time_triggers.BYR_REJECTION:
            feature_function = TimeFeatures._update_byr_rejection
        elif trigger_type == time_triggers.THREAD_EXPIRATION:
            feature_function = TimeFeatures._update_thread_expire
        elif trigger_type == time_triggers.SLR_REJECTION:
            pass  # no updates
        elif trigger_type == time_triggers.LSTG_EXPIRATION:
            del self.feats[ids['lstg_id']]  # just delete the lstg
            pass
        elif trigger_type == time_triggers.SALE:
            del self.feats[ids['lstg_id']]  # just delete the lstg
            pass
        else:
            raise NotImplementedError()
        for feat in consts.TIME_FEATS:
            feature_function(feat_dict=self.feats[ids['lstg_id']],
                             offer=offer, thread_id=ids['thread_id'])


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

    def promote(self, thread_id=None, value=None):
        """
        Updates the value of the entry with the given thread_id and maintains heap order
        
        :param value: updated value of thread_id. None implies
        value has already been updated and only the swapping must occur
        :param thread_id: integer giving the unique id of the target entry
        :return: Boolean for whether the value updated (false implies the promotion was actually a demotion)
        True implies value is the same as current value or more extreme (i.e. closer to the top element)
        :raises: RuntimeError if thread_id is not in the heap
        """
        if thread_id not in self:
            raise RuntimeError('may not promote non-existent element')
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

    def remove(self, thread_id=None):
        """
        Removes entry associated with given thread_id from the heap

        :param value: updated value of thread_id. None implies
        value has already been updated and only the swapping must occur
        :param thread_id: integer giving the unique id of the target entry
        :return: HeapEntry for the removed item
        :raises: RuntimeError if an entry with the thread_id doesn't exist in the heap
        """
        if thread_id not in self.index_map:
            raise RuntimeError("May not remove thread not present in heap")
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


class ExpirationHeap:
    """
    Attributes:
        time_heap: min FeatureHeap where values denote the time each element was last updated 
        feat_heap: FeatureHeap where values denote the value with each element is added to the ExpirationHeap
        size: number of elements in the expiration heap
        expiration: number of units of time until elements should be removed from the heap
        backups: dictionary mapping thread_id to deques of (time, value) tuples where, for each thread_id, each
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
        increment:
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
        self.contents[thread_id] += 1
        self.total += 1

    def remove(self, thread_id=None):
        """
        Remove contribution of some thread from the total

        :param thread_id: id of the current thread
        :return: NA
        """
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
        peek: counts how many elements the queue contains at a given time step, exlcudes count of entries
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
