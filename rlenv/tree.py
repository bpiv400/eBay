import numpy as np
import pandas as pd
import math

# deprecated but complete debugging anyway at some point
# and publish github gist for others (change date)


class Node:
    def __init__(self, **kwargs):
        '''
        Initializes tree leaf node with null children and key/value/parent
        from constructor, and 0 balance

        Kwargs:
            parent: Node reference to parent node in the tree
            date: date key associated with node (np.datetime)
            ix: index in numpy time array of associated value
        '''
        if 'parent' in kwargs:
            self.parent = kwargs['parent']
        else:
            self.__parent = None
        self.__height = 0
        # set key to date value in kwargs should be of type np.date
        self.__date = kwargs['date']
        # set index to value in kwargs (indicates index in time numpy array)
        self.__ix = kwargs['ix']
        # set left and right references to null
        self.__right = None
        self.__left = None

    def set_parent(self, par):
        self.__parent = par

    def get_parent(self):
        return self.__parent

    def is_bal(self):
        if self.is_leaf():
            return True
        else:
            # if one of the nodes is missing, the other node must be a leaf
            if self.__right is None:
                return self.__left.get_height() == 0
            elif self.__left is None:
                return self.__right.get_height() == 0
            else:
                # otherwise return whether the difference is less than or equal to 1
                return abs(self.__left.get_height() - self.__right.get_height()) <= 1

    def get_height(self):
        return self.__height

    def update_height(self):
        if self.is_leaf():
            self.__height = 0
        elif self.__left is None:
            self.__height = 1 + self.__right.get_height()
        elif self.__right is None:
            self.__height = 1 + self.__left.get_height()
        else:
            self.__height = 1 + max(self.__right.get_height(),
                                    self.__left.get_height())

    def get_bal(self):
        if self.is_leaf():
            return 0
        elif self.__right is None:
            return -1 - self.__left.get_height()
        elif self.__left is None:
            return 1 + self.__right.get_height()
        else:
            return self.__right.get_height() - self.__left.get_height()

    def get_right(self):
        return self.__right

    def get_left(self):
        return self.__left

    def get_ix(self):
        return self.__ix

    def get_date(self):
        return self.__date

    def set_date(self, date):
        self.__date = date

    def set_ix(self, ix):
        self.__ix = ix

    def set_left(self, newleft):
        self.__left = newleft

    def set_right(self, newright):
        self.__right = newright

    def is_leaf(self):
        '''
        Returns boolean indicating whether node is a leaf
        '''
        return (self.__right is None and self.__left is None)


class AVLTree:
    def __init__(self, **kwargs):
        '''
        Constructor initializes root and height
        '''
        self.size = 0
        self.root = None

    def insert(self, date, ix):
        if self.size == 0:
            self.root = Node(date=date, ix=ix)
            self.size += 1
        else:
            self.recurse_insert(self.root, date=date, ix=ix)

    def recurse_insert(self, curr, date, ix):
        # extract date
        currdate = curr.get_date()
        if currdate == date:
            # if date already in the tree with same, index do nothing
            if ix == curr.get_ix():
                pass
            # raise error if in the tree with different index
            else:
                raise ValueError("Should not insert anything already in tree")
        # look left
        if date < curr.get_date():
            # recurse as necessary
            if curr.get_left() is not None:
                inc = self.recurse_insert(curr.get_left(), date, ix)
            else:
                # insert node to the left
                insert = Node(date=date, ix=ix, parent=curr)
                curr.set_left(insert)
                curr.update_height()
                self.size += 1
                # return whether this increases height
                return curr.get_right() is None
        else:
            # if the right side is occupied
            if curr.get_right() is not None:
                # recurse
                inc = self.recurse_insert(curr.get_right(), date, ix)
            else:
                # or directly insert the node to the right
                insert = Node(date=date, ix=ix, parent=curr)
                curr.set_right(insert)
                curr.update_height()
                self.size += 1
                # and return whether height has increased
                return curr.get_left() is None

        # if depth increased
        if inc == 1:
            # update height and check whether balanced
            curr.update_height()
            if not curr.is_bal():
                self.rebalance_insert(curr, date)
                return 0
            # otherwise propogate depth increase up
            else:
                return 1
        else:
            return 0

    def left_rotate(self, par, child):
        # temporarily store child's left tree
        temp = child.get_left()
        # set new left tree to be the parent
        child.set_left(par)
        # set parent's new right child to be temp
        par.set_right(temp)
        # update height of parent and child
        par.update_height()
        child.update_height()

    def right_rotate(self, par, child):
        # temporarily store child's right tree
        temp = child.get_right()
        # set parent to be child's new right subtree
        child.set_right(par)
        # update parent of child
        child.set_par(par.get_parent())
        # update parent of parent
        par.set_parent(child)
        # update parent of child's former right child
        temp.set_parent(par)
        # update left child of parent with former right child of child
        par.set_left(temp)
        # update heights of parent and child
        par.update_height()
        child.update_height()

    def zigzig(self, node, right=True):
        '''
        Node gives the grandparent node in the zig-zig case
        '''
        if right:
            par = node.get_right()
            self.left_rotate(node, par)
        else:
            par = node.get_left()
            self.right_rotate(node, par)

    def zigzag(self, node, right=True):
        '''
        Node gives the grandparent node in the zigzag case
        Right gives whether we are rotating the right child or left
        '''
        if right:
            par = node.get_right()
            child = par.get_left()
            self.right_rotate(par, child)
            self.left_rotate(node, child)
        else:
            par = node.get_left()
            child = par.get_right()
            self.left_rotate(par, child)
            self.right_rotate(node, child)

    def get_max(self, node):
        if node.get_right() is None:
            return node
        else:
            return self.get_max(node.get_right())

    def get_predecessor(self, node):
        if node.get_left() is not None:
            return self.get_max(node.get_left())
        else:
            targ = node.get_date()
            while node.get_parent() is not None:
                node = node.get_parent()
                if node.get_date() < targ:
                    return node
            return None

    def recursive_delete(self, node, key):
        # base case
        if node is None:
            return None
        # found cases
        if key == node.get_date():
            # leaf node, we're chilling
            if node.is_leaf():
                return None
            # 1 child cases, promote other child
            elif node.get_left() is None:
                return node.get_right()
            elif node.get_right() is None:
                return node.get_left()
            else:
                # extract successor node
                pred = self.get_predecessor(node)
                # update value and key of current node
                node.set_date(pred.get_date())
                node.set_ix(pred.get_ix())
                # delete predecessor node
                node.set_left(self.recursive_delete(
                    node.get_left(), pred.get_date()))
        else:
            # recursive search cases
            if key < node.get_date():
                # left search
                node.set_left(self.recursive_delete(node.get_left(), key))
            else:
                # right search
                node.set_right(self.recursive_delete(node.get_right(), key))
        # update height, since nodes beneath have already been updated
        node.update_height()
        # check balance and fix as necessary
        if not node.is_bal():
            self.rebalance_delete(node)
        return node

    def rebalance_delete(self, node):
        # check whether we're doing a left rotation or a right one
        if node.get_bal() > 0:
            # implies the right side is heavy
            if node.get_right().get_bal() > 0:
                # if the node is inserted to ther ight of the right child
                self.zigzig(node, right=True)
            else:
                self.zigzag(node, right=True)
        else:
            # implies left side is heavy
            if node.get_left().get_bal() < 0:
                # if the node is inserted to the left of the left child
                self.zigzig(node, right=False)
            else:
                self.zigzag(node, right=False)

    def delete(self, key):
        # base case
        if self.size == 1:
            self.root = None
            self.size = 0
        # recursive case
        self.root = self.recursive_delete(self.root, key)

    def fuzzy_find(self, key):
        '''
        Finds the predecessor of the given key in the tree
        '''
        # basic error checking
        if self.root is None:
            raise ValueError("Tree is empty for some reason")
        self.recursive_fuzzy_find(self.root, key)

    def recursive_fuzzy_find(self, node, key):
        # if the current node is greater than the target
        if node.get_date() < key:
            # check whether we can recurse right
            if node.get_right() is None:
                # if not, return the current index
                return node.get_ix()
            else:
                return self.recursive_fuzzy_find(node.get_right(), key)
        # if we've gone too far to the left
        elif node.get_date() > key:
            if node.get_left() is None:
                # return the value of the parent
                out = self.get_predecessor(node)
                if out is None:
                    return None
                else:
                    return out.get_ix()
            else:
                return self.recursive_fuzzy_find(node.get_left(), key)

    def rebalance_insert(self, node, insert_key):
        # check whether we're doing a left rotation or a right one
        if insert_key > node.get_date():
            # implies the right side is heavy
            if insert_key > node.get_right().get_date():
                # if the node is inserted to ther ight of the right child
                self.zigzig(node, right=True)
            else:
                self.zigzag(node, right=True)
        else:
            # implies left side is heavy
            if insert_key < node.get_left().get_date():
                # if the node is inserted to the left of the left child
                self.zigzig(node, right=False)
            else:
                self.zigzag(node, right=False)
