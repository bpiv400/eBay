from rlenv.TimeFeatures import SmallHeap, HeapEntry
import pytest


@pytest.fixture
def heap():
    return SmallHeap()


def test_empty(heap):
    assert heap.peek(exclude=0) == 0
    assert heap.peek() == 0
    assert len(heap.contents) == 0


def test_push_singleton(heap):
    heap.promote(thread_id=0, value=.5)
    assert heap.peek(exclude=0) == 0
    assert heap.peek(exclude=1) == .5
    assert heap.peek() == .5
    assert len(heap.contents) == 1


def test_promote_singleton(heap):
    heap.promote(thread_id=0, value=.5)
    heap.promote(thread_id=0, value=.7)
    assert heap.peek(exclude=0) == 0
    assert heap.peek(exclude=1) == .7
    assert heap.peek() == .7
    assert len(heap.contents) == 1


def test_demote_singleton(heap):
    heap.promote(thread_id=0, value=.5)
    heap.promote(thread_id=0, value=.4)
    assert len(heap.contents) == 1
    assert heap.peek(exclude=0) == 0
    assert heap.peek(exclude=1) == .5
    assert heap.peek() == .5


def test_push_full(heap):
    heap.promote(thread_id=0, value=.5)
    heap.promote(thread_id=1, value=.8)
    assert len(heap.contents) == 2
    assert heap.peek() == .8
    assert heap.peek(exclude=1) == .5
    assert heap.peek(exclude=0) == .8


def test_promote_full(heap):
    heap.promote(thread_id=0, value=.5)
    heap.promote(thread_id=1, value=.8)
    heap.promote(thread_id=0, value=.6)
    assert len(heap.contents) == 2
    assert heap.peek() == .8
    assert heap.peek(exclude=1) == .6
    assert heap.peek(exclude=0) == .8
    heap.promote(thread_id=1, value=.9)
    assert len(heap.contents) == 2
    assert heap.peek() == .9
    assert heap.peek(exclude=1) == .6
    assert heap.peek(exclude=0) == .9
    heap.promote(thread_id=0, value=.95)
    assert len(heap.contents) == 2
    assert heap.peek() == .95
    assert heap.peek(exclude=1) == .95
    assert heap.peek(exclude=0) == .9


def test_demote_full(heap):
    heap.promote(thread_id=0, value=.5)
    heap.promote(thread_id=1, value=.8)
    heap.promote(thread_id=0, value=.4)
    assert len(heap.contents) == 2
    assert heap.peek() == .8
    assert heap.peek(exclude=1) == .5
    assert heap.peek(exclude=0) == .8
    heap.promote(thread_id=1, value=.7)
    assert len(heap.contents) == 2
    assert heap.peek() == .8
    assert heap.peek(exclude=1) == .5
    assert heap.peek(exclude=0) == .8


def test_push_median(heap):
    heap.promote(thread_id=0, value=.5)
    heap.promote(thread_id=1, value=.8)
    heap.promote(thread_id=2, value=.6)
    assert len(heap.contents) == 2
    assert heap.peek() == .8
    assert heap.peek(exclude=1) == .6
    assert heap.peek(exclude=0) == .8
    assert heap.peek(exclude=2) == .8


def test_push_max(heap):
    heap.promote(thread_id=0, value=.5)
    heap.promote(thread_id=1, value=.8)
    heap.promote(thread_id=2, value=.9)
    assert len(heap.contents) == 2
    assert heap.peek() == .9
    assert heap.peek(exclude=1) == .9
    assert heap.peek(exclude=0) == .9
    assert heap.peek(exclude=2) == .8


def test_push_full_min(heap):
    heap.promote(thread_id=0, value=.5)
    heap.promote(thread_id=1, value=.8)
    heap.promote(thread_id=2, value=.4)
    assert len(heap.contents) == 2
    assert heap.peek() == .8
    assert heap.peek(exclude=1) == .5
    assert heap.peek(exclude=0) == .8
    assert heap.peek(exclude=2) == .8
