from deprecated.TimeFeatures import FeatureHeap, HeapEntry
import pytest


def index_map_correct(heap):
    """
    Ensures the index map is correct

    """
    for i, entry in enumerate(heap.heap):
        assert heap.index_map[entry.thread_id] == i
    assert heap.size == len(heap.heap)
    assert heap.size == len(heap.index_map)


@pytest.fixture
def make_heap():
    min_heap = FeatureHeap(min_heap=True)
    max_heap = FeatureHeap(min_heap=False)
    return max_heap, min_heap


def test_heap_entry_order(make_heap):
    a = HeapEntry(thread_id=4, value=20)
    b = HeapEntry(thread_id=25, value=4)
    assert b < a
    assert a > b
    assert a != b


def test_push_insert_in_order(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=20)
    max_heap.push(thread_id=1, value=10)
    max_heap.push(thread_id=2, value=5)
    max_heap.push(thread_id=3, value=0)

    min_heap.push(thread_id=3, value=0)
    min_heap.push(thread_id=2, value=5)
    min_heap.push(thread_id=1, value=10)
    min_heap.push(thread_id=0, value=20)

    max_actual = [20, 10, 5, 0]
    min_actual = [0, 5, 10, 20]

    assert all([a == b.value for a, b in zip(max_actual, max_heap.heap)])
    assert all([a == b.value for a, b in zip(min_actual, min_heap.heap)])
    assert all([a == min_heap.index_map[idx] for a, idx in zip(list(range(4)), [3, 2, 1, 0])])
    assert all([a == max_heap.index_map[idx] for a, idx in zip(list(range(4)), [0, 1, 2, 3])])


def test_push_reverse_order(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=0)
    max_heap.push(thread_id=1, value=5)
    max_heap.push(thread_id=2, value=10)
    max_heap.push(thread_id=3, value=20)

    min_heap.push(thread_id=3, value=20)
    min_heap.push(thread_id=2, value=10)
    min_heap.push(thread_id=1, value=5)
    min_heap.push(thread_id=0, value=0)

    assert min_heap.peak(exclude=None) == 0
    assert max_heap.peak(exclude=None) == 20

    index_map_correct(min_heap)
    index_map_correct(max_heap)

    max_expect = [20, 10, 5, 0]
    min_expect = [0, 5, 10, 20]

    for i in range(4):
        a = max_heap.pop()
        assert a.value == max_expect[i]
        assert a.thread_id not in max_heap
        assert max_heap.size == (3 - i)
        index_map_correct(max_heap)
        a = min_heap.pop()
        assert a.value == min_expect[i]
        assert a.thread_id not in min_heap
        assert min_heap.size == (3 - i)


def test_demote_leaf_still_leaf(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=0)
    max_heap.push(thread_id=1, value=15)
    max_heap.push(thread_id=2, value=2)
    max_heap.push(thread_id=3, value=9)
    max_heap.push(thread_id=4, value=50)

    min_heap.push(thread_id=0, value=0)
    min_heap.push(thread_id=1, value=15)
    min_heap.push(thread_id=2, value=2)
    min_heap.push(thread_id=3, value=9)
    min_heap.push(thread_id=4, value=-4)

    max_values = [-1, 15, 2, 9, 50]
    min_values = [0, 20, 2, 9, -4]

    assert min_heap.size == 5
    assert max_heap.size == 5

    max_expect = sorted(max_values)[::-1]
    min_expect = sorted(min_values)

    assert max_heap.demote(value=-1, thread_id=0)
    assert min_heap.demote(value=20, thread_id=1)
    index_map_correct(max_heap)
    index_map_correct(min_heap)

    for i in range(5):
        a = max_heap.pop()
        assert a.value == max_expect[i]
        assert a.thread_id not in max_heap
        assert max_heap.size == (4 - i)
        index_map_correct(max_heap)
        a = min_heap.pop()
        assert a.value == min_expect[i]
        assert a.thread_id not in min_heap
        assert min_heap.size == (4 - i)
        index_map_correct(min_heap)


def test_demote_root_still_root(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=0)
    max_heap.push(thread_id=1, value=15)
    max_heap.push(thread_id=2, value=2)
    max_heap.push(thread_id=3, value=9)
    max_heap.push(thread_id=4, value=50)

    min_heap.push(thread_id=0, value=0)
    min_heap.push(thread_id=1, value=15)
    min_heap.push(thread_id=2, value=2)
    min_heap.push(thread_id=3, value=9)
    min_heap.push(thread_id=4, value=-4)

    max_values = [0, 15, 2, 9, 45]
    min_values = [0, 15, 2, 9, -3]

    assert min_heap.size == 5
    assert max_heap.size == 5

    max_expect = sorted(max_values)[::-1]
    min_expect = sorted(min_values)

    assert max_heap.demote(value=45, thread_id=4)
    assert min_heap.demote(value=-3, thread_id=4)
    index_map_correct(max_heap)
    index_map_correct(min_heap)

    for i in range(5):
        a = max_heap.pop()
        assert a.value == max_expect[i]
        assert a.thread_id not in max_heap
        assert max_heap.size == (4 - i)
        index_map_correct(max_heap)
        a = min_heap.pop()
        assert a.value == min_expect[i]
        assert a.thread_id not in min_heap
        assert min_heap.size == (4 - i)
        index_map_correct(min_heap)


def test_demote_root_to_middle(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=0)
    max_heap.push(thread_id=1, value=15)
    max_heap.push(thread_id=2, value=2)
    max_heap.push(thread_id=3, value=9)
    max_heap.push(thread_id=4, value=50)

    min_heap.push(thread_id=0, value=0)
    min_heap.push(thread_id=1, value=15)
    min_heap.push(thread_id=2, value=2)
    min_heap.push(thread_id=3, value=9)
    min_heap.push(thread_id=4, value=-4)

    max_values = [0, 15, 2, 9, 10]
    min_values = [0, 15, 2, 9, 5]

    assert min_heap.size == 5
    assert max_heap.size == 5

    max_expect = sorted(max_values)[::-1]
    min_expect = sorted(min_values)

    assert max_heap.demote(value=10, thread_id=4)
    assert min_heap.demote(value=5, thread_id=4)
    index_map_correct(max_heap)
    index_map_correct(min_heap)

    for i in range(5):
        a = max_heap.pop()
        assert a.value == max_expect[i]
        assert a.thread_id not in max_heap
        assert max_heap.size == (4 - i)
        index_map_correct(max_heap)
        a = min_heap.pop()
        assert a.value == min_expect[i]
        assert a.thread_id not in min_heap
        assert min_heap.size == (4 - i)
        index_map_correct(min_heap)


def test_demote_root_to_last(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=0)
    max_heap.push(thread_id=1, value=15)
    max_heap.push(thread_id=2, value=2)
    max_heap.push(thread_id=3, value=9)
    max_heap.push(thread_id=4, value=50)

    min_heap.push(thread_id=0, value=0)
    min_heap.push(thread_id=1, value=15)
    min_heap.push(thread_id=2, value=2)
    min_heap.push(thread_id=3, value=9)
    min_heap.push(thread_id=4, value=-4)

    max_values = [0, 15, 2, 9, -4]
    min_values = [0, 15, 2, 9, 50]

    assert min_heap.size == 5
    assert max_heap.size == 5

    max_expect = sorted(max_values)[::-1]
    min_expect = sorted(min_values)

    max_heap.demote(value=-4, thread_id=4)
    min_heap.demote(value=50, thread_id=4)
    index_map_correct(max_heap)
    index_map_correct(min_heap)

    for i in range(5):
        a = max_heap.pop()
        assert a.value == max_expect[i]
        assert a.thread_id not in max_heap
        assert max_heap.size == (4 - i)
        index_map_correct(max_heap)
        a = min_heap.pop()
        assert a.value == min_expect[i]
        assert a.thread_id not in min_heap
        assert min_heap.size == (4 - i)
        index_map_correct(min_heap)


def test_promote_leaf_to_root(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=0)
    max_heap.push(thread_id=1, value=15)
    max_heap.push(thread_id=2, value=2)
    max_heap.push(thread_id=3, value=9)
    max_heap.push(thread_id=4, value=50)

    min_heap.push(thread_id=0, value=0)
    min_heap.push(thread_id=1, value=15)
    min_heap.push(thread_id=2, value=2)
    min_heap.push(thread_id=3, value=9)
    min_heap.push(thread_id=4, value=50)

    max_values = [55, 15, 2, 9, 50]
    min_values = [0, 15, 2, 9, -4]

    assert min_heap.size == 5
    assert max_heap.size == 5

    max_expect = sorted(max_values)[::-1]
    min_expect = sorted(min_values)

    max_heap.promote(value=55, thread_id=0)
    min_heap.promote(value=-4, thread_id=4)
    index_map_correct(max_heap)
    index_map_correct(min_heap)

    for i in range(5):
        a = max_heap.pop()
        assert a.value == max_expect[i]
        assert a.thread_id not in max_heap
        assert max_heap.size == (4 - i)
        index_map_correct(max_heap)
        a = min_heap.pop()
        assert a.value == min_expect[i]
        assert a.thread_id not in min_heap
        assert min_heap.size == (4 - i)
        index_map_correct(min_heap)


def test_promote_leaf_to_leaf(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=0)
    max_heap.push(thread_id=1, value=15)
    max_heap.push(thread_id=2, value=2)
    max_heap.push(thread_id=3, value=9)
    max_heap.push(thread_id=4, value=50)

    min_heap.push(thread_id=0, value=0)
    min_heap.push(thread_id=1, value=15)
    min_heap.push(thread_id=2, value=2)
    min_heap.push(thread_id=3, value=9)
    min_heap.push(thread_id=4, value=50)

    max_values = [1, 15, 2, 9, 50]
    min_values = [0, 15, 2, 9, 45]

    assert min_heap.size == 5
    assert max_heap.size == 5

    max_expect = sorted(max_values)[::-1]
    min_expect = sorted(min_values)

    max_heap.promote(value=1, thread_id=0)
    min_heap.promote(value=45, thread_id=4)
    index_map_correct(max_heap)
    index_map_correct(min_heap)

    for i in range(5):
        a = max_heap.pop()
        assert a.value == max_expect[i]
        assert a.thread_id not in max_heap
        assert max_heap.size == (4 - i)
        index_map_correct(max_heap)
        a = min_heap.pop()
        assert a.value == min_expect[i]
        assert a.thread_id not in min_heap
        assert min_heap.size == (4 - i)
        index_map_correct(min_heap)


def test_promote_leaf_to_middle(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=0)
    max_heap.push(thread_id=1, value=15)
    max_heap.push(thread_id=2, value=2)
    max_heap.push(thread_id=3, value=9)
    max_heap.push(thread_id=4, value=50)

    min_heap.push(thread_id=0, value=0)
    min_heap.push(thread_id=1, value=15)
    min_heap.push(thread_id=2, value=2)
    min_heap.push(thread_id=3, value=9)
    min_heap.push(thread_id=4, value=50)

    max_values = [45, 15, 2, 9, 50]
    min_values = [0, 15, 2, 9, 1]

    assert min_heap.size == 5
    assert max_heap.size == 5

    max_expect = sorted(max_values)[::-1]
    min_expect = sorted(min_values)

    max_heap.promote(value=45, thread_id=0)
    min_heap.promote(value=1, thread_id=4)
    index_map_correct(max_heap)
    index_map_correct(min_heap)

    for i in range(5):
        a = max_heap.pop()
        assert a.value == max_expect[i]
        assert a.thread_id not in max_heap
        assert max_heap.size == (4 - i)
        index_map_correct(max_heap)
        a = min_heap.pop()
        assert a.value == min_expect[i]
        assert a.thread_id not in min_heap
        assert min_heap.size == (4 - i)
        index_map_correct(min_heap)


def test_push_many_random_order(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=0)
    max_heap.push(thread_id=1, value=15)
    max_heap.push(thread_id=2, value=2)
    max_heap.push(thread_id=3, value=9)
    max_heap.push(thread_id=4, value=50)
    max_heap.push(thread_id=5, value=26)
    max_heap.push(thread_id=6, value=28)
    max_heap.push(thread_id=7, value=1)

    min_heap.push(thread_id=0, value=0)
    min_heap.push(thread_id=1, value=15)
    min_heap.push(thread_id=2, value=2)
    min_heap.push(thread_id=3, value=9)
    min_heap.push(thread_id=4, value=50)
    min_heap.push(thread_id=5, value=26)
    min_heap.push(thread_id=6, value=28)
    min_heap.push(thread_id=7, value=1)

    index_map_correct(min_heap)
    index_map_correct(max_heap)

    values = [0, 15, 2, 9, 50, 26, 28, 1]
    min_expect = sorted(values)
    max_expect = min_expect[::-1]

    for i in range(8):
        a = max_heap.pop()
        assert a.value == max_expect[i]
        assert a.thread_id not in max_heap
        assert max_heap.size == (7 - i)
        index_map_correct(max_heap)
        a = min_heap.pop()
        assert a.value == min_expect[i]
        assert a.thread_id not in min_heap
        assert min_heap.size == (7 - i)
        index_map_correct(min_heap)


def test_demote_middle_to_leaf(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=0)
    max_heap.push(thread_id=1, value=15)
    max_heap.push(thread_id=2, value=2)
    max_heap.push(thread_id=3, value=9)
    max_heap.push(thread_id=4, value=50)
    max_heap.push(thread_id=5, value=26)
    max_heap.push(thread_id=6, value=28)
    max_heap.push(thread_id=7, value=1)

    min_heap.push(thread_id=0, value=0)
    min_heap.push(thread_id=1, value=15)
    min_heap.push(thread_id=2, value=2)
    min_heap.push(thread_id=3, value=9)
    min_heap.push(thread_id=4, value=50)
    min_heap.push(thread_id=5, value=26)
    min_heap.push(thread_id=6, value=28)
    min_heap.push(thread_id=7, value=1)

    index_map_correct(min_heap)
    index_map_correct(max_heap)

    max_heap.demote(thread_id=6, value=-1)
    min_heap.demote(thread_id=7, value=52)
    max_values = [0, 15, 2, 9, 50, 26, 1, -1]
    min_values = [0, 15, 2, 9, 50, 26, 28, 52]
    min_expect = sorted(min_values)
    max_expect = sorted(max_values)[::-1]
    index_map_correct(max_heap)
    index_map_correct(min_heap)

    for i in range(8):
        a = max_heap.pop()
        assert a.value == max_expect[i]
        assert a.thread_id not in max_heap
        assert max_heap.size == (7 - i)
        index_map_correct(max_heap)
        a = min_heap.pop()
        assert a.value == min_expect[i]
        assert a.thread_id not in min_heap
        assert min_heap.size == (7 - i)
        index_map_correct(min_heap)


def test_promote_middle_to_root(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=0)
    max_heap.push(thread_id=1, value=15)
    max_heap.push(thread_id=2, value=2)
    max_heap.push(thread_id=3, value=9)
    max_heap.push(thread_id=4, value=50)
    max_heap.push(thread_id=5, value=26)
    max_heap.push(thread_id=6, value=28)
    max_heap.push(thread_id=7, value=1)

    min_heap.push(thread_id=0, value=0)
    min_heap.push(thread_id=1, value=15)
    min_heap.push(thread_id=2, value=2)
    min_heap.push(thread_id=3, value=9)
    min_heap.push(thread_id=4, value=50)
    min_heap.push(thread_id=5, value=26)
    min_heap.push(thread_id=6, value=28)
    min_heap.push(thread_id=7, value=1)

    index_map_correct(min_heap)
    index_map_correct(max_heap)

    max_heap.promote(thread_id=6, value=55)
    min_heap.promote(thread_id=7, value=-1)
    max_values = [0, 15, 2, 9, 50, 26, 1, 55]
    min_values = [0, 15, 2, 9, 50, 26, 28, -1]
    min_expect = sorted(min_values)
    max_expect = sorted(max_values)[::-1]
    index_map_correct(max_heap)
    index_map_correct(min_heap)

    for i in range(8):
        a = max_heap.pop()
        assert a.value == max_expect[i]
        assert a.thread_id not in max_heap
        assert max_heap.size == (7 - i)
        index_map_correct(max_heap)
        a = min_heap.pop()
        assert a.value == min_expect[i]
        assert a.thread_id not in min_heap
        assert min_heap.size == (7 - i)
        index_map_correct(min_heap)


def test_promote_middle_to_root(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=0)
    max_heap.push(thread_id=1, value=15)
    max_heap.push(thread_id=2, value=2)
    max_heap.push(thread_id=3, value=9)
    max_heap.push(thread_id=4, value=50)
    max_heap.push(thread_id=5, value=26)
    max_heap.push(thread_id=6, value=28)
    max_heap.push(thread_id=7, value=1)

    min_heap.push(thread_id=0, value=0)
    min_heap.push(thread_id=1, value=15)
    min_heap.push(thread_id=2, value=2)
    min_heap.push(thread_id=3, value=9)
    min_heap.push(thread_id=4, value=50)
    min_heap.push(thread_id=5, value=26)
    min_heap.push(thread_id=6, value=28)
    min_heap.push(thread_id=7, value=1)

    index_map_correct(min_heap)
    index_map_correct(max_heap)

    max_heap.promote(thread_id=6, value=29)
    min_heap.promote(thread_id=7, value=.5)
    max_values = [0, 15, 2, 9, 50, 26, 1, 29]
    min_values = [0, 15, 2, 9, 50, 26, 28, .5]
    min_expect = sorted(min_values)
    max_expect = sorted(max_values)[::-1]
    index_map_correct(max_heap)
    index_map_correct(min_heap)

    for i in range(8):
        a = max_heap.pop()
        assert a.value == max_expect[i]
        assert a.thread_id not in max_heap
        assert max_heap.size == (7 - i)
        index_map_correct(max_heap)
        a = min_heap.pop()
        assert a.value == min_expect[i]
        assert a.thread_id not in min_heap
        assert min_heap.size == (7 - i)
        index_map_correct(min_heap)


def test_promote_for_demote(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=0)
    min_heap.push(thread_id=0, value=0)

    assert not max_heap.promote(thread_id=0, value=-1)
    assert not min_heap.promote(thread_id=0, value=1)
    assert max_heap.promote(thread_id=0, value=0)
    assert min_heap.promote(thread_id=0, value=0)


def test_promote_none(make_heap):
    max_heap, min_heap = make_heap
    with pytest.raises(RuntimeError):
        max_heap.promote(thread_id=None, value=5)

    with pytest.raises(RuntimeError):
        min_heap.promote(thread_id=None, value=-5)


def test_remove_singleton(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=0)
    min_heap.push(thread_id=0, value=0)
    max_heap.remove(thread_id=0)
    min_heap.remove(thread_id=0)
    assert 0 not in max_heap
    assert 0 not in min_heap
    assert max_heap.size == min_heap.size
    assert max_heap.size == 0

def test_remove_root_2_children(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=0)
    max_heap.push(thread_id=1, value=5)
    max_heap.push(thread_id=2, value=10)

    min_heap.push(thread_id=0, value=0)
    min_heap.push(thread_id=1, value=5)
    min_heap.push(thread_id=2, value=10)

    min_expect = [5, 10]
    max_expect = [5, 0]

    index_map_correct(max_heap)
    index_map_correct(min_heap)

    max_heap.remove(thread_id=2)
    min_heap.remove(thread_id=0)

    index_map_correct(max_heap)
    index_map_correct(min_heap)

    for i in range(2):
        a = max_heap.pop()
        assert a.value == max_expect[i]
        assert a.thread_id not in max_heap
        assert max_heap.size == (1 - i)
        index_map_correct(max_heap)
        a = min_heap.pop()
        assert a.value == min_expect[i]
        assert a.thread_id not in min_heap
        assert min_heap.size == (1 - i)
        index_map_correct(min_heap)


def test_remove_root_1_child(make_heap):
    max_heap, min_heap = make_heap

    max_heap.push(thread_id=0, value=0)
    max_heap.push(thread_id=1, value=5)

    min_heap.push(thread_id=0, value=0)
    min_heap.push(thread_id=1, value=5)

    min_expect = [5]
    max_expect = [0]

    index_map_correct(max_heap)
    index_map_correct(min_heap)

    max_heap.remove(thread_id=1)
    min_heap.remove(thread_id=0)

    index_map_correct(max_heap)
    index_map_correct(min_heap)

    for i in range(1):
        a = max_heap.pop()
        assert a.value == max_expect[i]
        assert a.thread_id not in max_heap
        assert max_heap.size == (0 - i)
        index_map_correct(max_heap)
        a = min_heap.pop()
        assert a.value == min_expect[i]
        assert a.thread_id not in min_heap
        assert min_heap.size == (0 - i)
        index_map_correct(min_heap)
    

def test_remove_leaf(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=0)
    max_heap.push(thread_id=1, value=15)
    max_heap.push(thread_id=2, value=2)
    max_heap.push(thread_id=3, value=9)
    max_heap.push(thread_id=4, value=50)
    max_heap.push(thread_id=5, value=26)
    max_heap.push(thread_id=6, value=28)
    max_heap.push(thread_id=7, value=1)

    min_heap.push(thread_id=0, value=0)
    min_heap.push(thread_id=1, value=15)
    min_heap.push(thread_id=2, value=2)
    min_heap.push(thread_id=3, value=9)
    min_heap.push(thread_id=4, value=50)
    min_heap.push(thread_id=5, value=26)
    min_heap.push(thread_id=6, value=28)
    min_heap.push(thread_id=7, value=1)

    index_map_correct(min_heap)
    index_map_correct(max_heap)

    max_heap.remove(thread_id=0)
    min_heap.remove(thread_id=4)
    max_values = [15, 2, 9, 50, 26, 1, 28]
    min_values = [0, 15, 2, 9, 26, 28, 1]
    min_expect = sorted(min_values)
    max_expect = sorted(max_values)[::-1]
    index_map_correct(max_heap)
    index_map_correct(min_heap)

    for i in range(7):
        a = max_heap.pop()
        assert a.value == max_expect[i]
        assert a.thread_id not in max_heap
        assert max_heap.size == (6 - i)
        index_map_correct(max_heap)
        a = min_heap.pop()
        assert a.value == min_expect[i]
        assert a.thread_id not in min_heap
        assert min_heap.size == (6 - i)
        index_map_correct(min_heap)


def test_remove_middle(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=0)
    max_heap.push(thread_id=1, value=15)
    max_heap.push(thread_id=2, value=2)
    max_heap.push(thread_id=3, value=9)
    max_heap.push(thread_id=4, value=50)
    max_heap.push(thread_id=5, value=26)
    max_heap.push(thread_id=6, value=28)
    max_heap.push(thread_id=7, value=1)

    min_heap.push(thread_id=0, value=0)
    min_heap.push(thread_id=1, value=15)
    min_heap.push(thread_id=2, value=2)
    min_heap.push(thread_id=3, value=9)
    min_heap.push(thread_id=4, value=50)
    min_heap.push(thread_id=5, value=26)
    min_heap.push(thread_id=6, value=28)
    min_heap.push(thread_id=7, value=1)

    index_map_correct(min_heap)
    index_map_correct(max_heap)

    max_heap.remove(thread_id=6)
    min_heap.remove(thread_id=7)
    max_values = [15, 2, 9, 50, 26, 1, 0]
    min_values = [0, 15, 2, 9, 26, 28, 50]
    min_expect = sorted(min_values)
    max_expect = sorted(max_values)[::-1]
    index_map_correct(max_heap)
    index_map_correct(min_heap)

    for i in range(7):
        a = max_heap.pop()
        assert a.value == max_expect[i]
        assert a.thread_id not in max_heap
        assert max_heap.size == (6 - i)
        index_map_correct(max_heap)
        a = min_heap.pop()
        assert a.value == min_expect[i]
        assert a.thread_id not in min_heap
        assert min_heap.size == (6 - i)
        index_map_correct(min_heap)


def test_remove_root_large(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=0)
    max_heap.push(thread_id=1, value=15)
    max_heap.push(thread_id=2, value=2)
    max_heap.push(thread_id=3, value=9)
    max_heap.push(thread_id=4, value=50)
    max_heap.push(thread_id=5, value=26)
    max_heap.push(thread_id=6, value=28)
    max_heap.push(thread_id=7, value=1)

    min_heap.push(thread_id=0, value=0)
    min_heap.push(thread_id=1, value=15)
    min_heap.push(thread_id=2, value=2)
    min_heap.push(thread_id=3, value=9)
    min_heap.push(thread_id=4, value=50)
    min_heap.push(thread_id=5, value=26)
    min_heap.push(thread_id=6, value=28)
    min_heap.push(thread_id=7, value=1)

    index_map_correct(min_heap)
    index_map_correct(max_heap)

    max_heap.remove(thread_id=4)
    min_heap.remove(thread_id=0)
    max_values = [15, 2, 9, 26, 1, 28, 0]
    min_values = [50, 15, 2, 9, 26, 28, 1]
    min_expect = sorted(min_values)
    max_expect = sorted(max_values)[::-1]
    index_map_correct(max_heap)
    index_map_correct(min_heap)

    for i in range(7):
        a = max_heap.pop()
        assert a.value == max_expect[i]
        assert a.thread_id not in max_heap
        assert max_heap.size == (6 - i)
        index_map_correct(max_heap)
        a = min_heap.pop()
        assert a.value == min_expect[i]
        assert a.thread_id not in min_heap
        assert min_heap.size == (6 - i)
        index_map_correct(min_heap)

def test_promote_not_in_heap(make_heap):
    max_heap, min_heap = make_heap
    with pytest.raises(RuntimeError):
        max_heap.promote(thread_id=0, value=5)

    with pytest.raises(RuntimeError):
        min_heap.promote(thread_id=0, value=-5)


def test_promote_singleton(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=0)
    min_heap.push(thread_id=0, value=0)
    max_heap.promote(thread_id=0, value=5)
    min_heap.promote(thread_id=0, value=-5)

    min_expect = [-5]
    max_expect = [5]

    index_map_correct(max_heap)
    index_map_correct(min_heap)

    for i in range(1):
        a = max_heap.pop()
        assert a.value == max_expect[i]
        assert a.thread_id not in max_heap
        assert max_heap.size == (0 - i)
        index_map_correct(max_heap)
        a = min_heap.pop()
        assert a.value == min_expect[i]
        assert a.thread_id not in min_heap
        assert min_heap.size == (0 - i)
        index_map_correct(min_heap)


def test_demote_middle_to_middle(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=0)
    max_heap.push(thread_id=1, value=15)
    max_heap.push(thread_id=2, value=2)
    max_heap.push(thread_id=3, value=9)
    max_heap.push(thread_id=4, value=50)
    max_heap.push(thread_id=5, value=26)
    max_heap.push(thread_id=6, value=28)
    max_heap.push(thread_id=7, value=1)

    min_heap.push(thread_id=0, value=0)
    min_heap.push(thread_id=1, value=15)
    min_heap.push(thread_id=2, value=2)
    min_heap.push(thread_id=3, value=9)
    min_heap.push(thread_id=4, value=50)
    min_heap.push(thread_id=5, value=26)
    min_heap.push(thread_id=6, value=28)
    min_heap.push(thread_id=7, value=1)

    index_map_correct(min_heap)
    index_map_correct(max_heap)

    max_heap.demote(thread_id=6, value=27)
    min_heap.demote(thread_id=7, value=1.5)
    max_values = [0, 15, 2, 9, 50, 26, 1, 27]
    min_values = [0, 15, 2, 9, 50, 26, 28, 1.5]
    min_expect = sorted(min_values)
    max_expect = sorted(max_values)[::-1]
    index_map_correct(max_heap)
    index_map_correct(min_heap)

    for i in range(8):
        a = max_heap.pop()
        assert a.value == max_expect[i]
        assert a.thread_id not in max_heap
        assert max_heap.size == (7 - i)
        index_map_correct(max_heap)
        a = min_heap.pop()
        assert a.value == min_expect[i]
        assert a.thread_id not in min_heap
        assert min_heap.size == (7 - i)
        index_map_correct(min_heap)


def test_demote_singleton(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=4, value=2)
    min_heap.push(thread_id=4, value=2)

    max_heap.demote(thread_id=4, value=1)
    min_heap.demote(thread_id=4, value=3)
    max_values = [1]
    min_values = [3]
    min_expect = sorted(min_values)
    max_expect = sorted(max_values)[::-1]
    index_map_correct(max_heap)
    index_map_correct(min_heap)

    for i in range(1):
        a = max_heap.pop()
        assert a.value == max_expect[i]
        assert a.thread_id not in max_heap
        assert max_heap.size == (0 - i)
        index_map_correct(max_heap)
        a = min_heap.pop()
        assert a.value == min_expect[i]
        assert a.thread_id not in min_heap
        assert min_heap.size == (0 - i)
        index_map_correct(min_heap)


def test_demote_not_in(make_heap):
    max_heap, min_heap = make_heap
    with pytest.raises(RuntimeError):
        max_heap.demote(value=1, thread_id=5)

    with pytest.raises(RuntimeError):
        min_heap.demote(value=1, thread_id=5)


def test_demote_for_promote(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=4, value=2)
    min_heap.push(thread_id=4, value=2)
    with pytest.raises(RuntimeError):
        max_heap.demote(thread_id=4, value=3)
    with pytest.raises(RuntimeError):
        min_heap.demote(thread_id=4, value=1)


def test_demote_none(make_heap):
    max_heap, min_heap = make_heap
    with pytest.raises(RuntimeError):
        max_heap.demote(value=1, thread_id=None)


def test_demote_one_child(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=4, value=2)
    max_heap.push(thread_id=5, value=12)

    min_heap.push(thread_id=4, value=2)
    min_heap.push(thread_id=5, value=12)

    max_heap.demote(thread_id=5, value=1)
    min_heap.demote(thread_id=4, value=15)
    max_values = [2, 1]
    min_values = [12, 15]
    min_expect = sorted(min_values)
    max_expect = sorted(max_values)[::-1]
    index_map_correct(max_heap)
    index_map_correct(min_heap)

    for i in range(2):
        a = max_heap.pop()
        assert a.value == max_expect[i]
        assert a.thread_id not in max_heap
        assert max_heap.size == (1 - i)
        index_map_correct(max_heap)
        a = min_heap.pop()
        assert a.value == min_expect[i]
        assert a.thread_id not in min_heap
        assert min_heap.size == (1 - i)
        index_map_correct(min_heap)


def test_push_twice(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=4, value=2)
    min_heap.push(thread_id=4, value=2)
    with pytest.raises(RuntimeError):
        max_heap.push(thread_id=4)

    with pytest.raises(RuntimeError):
        min_heap.push(thread_id=4)


def test_push_none_thread_id(make_heap):
    max_heap, _ = make_heap
    with pytest.raises(RuntimeError):
        max_heap.push(value=5)


def test_push_none_value(make_heap):
    max_heap, _ = make_heap
    with pytest.raises(RuntimeError):
        max_heap.push(thread_id=5)


def test_peak_no_exclude(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=0)
    max_heap.push(thread_id=1, value=15)
    max_heap.push(thread_id=2, value=2)
    max_heap.push(thread_id=3, value=9)
    max_heap.push(thread_id=4, value=50)
    max_heap.push(thread_id=5, value=26)
    max_heap.push(thread_id=6, value=28)
    max_heap.push(thread_id=7, value=1)

    min_heap.push(thread_id=0, value=0)
    min_heap.push(thread_id=1, value=15)
    min_heap.push(thread_id=2, value=2)
    min_heap.push(thread_id=3, value=9)
    min_heap.push(thread_id=4, value=50)
    min_heap.push(thread_id=5, value=26)
    min_heap.push(thread_id=6, value=28)
    min_heap.push(thread_id=7, value=1)

    assert min_heap.peak(exclude=None) == 0
    assert max_heap.peak(exclude=None) == 50


def test_peak_exclude_root_two_children_large(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=0)
    max_heap.push(thread_id=1, value=15)
    max_heap.push(thread_id=2, value=2)
    max_heap.push(thread_id=3, value=9)
    max_heap.push(thread_id=4, value=50)
    max_heap.push(thread_id=5, value=26)
    max_heap.push(thread_id=6, value=28)
    max_heap.push(thread_id=7, value=1)

    min_heap.push(thread_id=0, value=0)
    min_heap.push(thread_id=1, value=15)
    min_heap.push(thread_id=2, value=2)
    min_heap.push(thread_id=3, value=9)
    min_heap.push(thread_id=4, value=50)
    min_heap.push(thread_id=5, value=26)
    min_heap.push(thread_id=6, value=28)
    min_heap.push(thread_id=7, value=1)

    assert min_heap.peak(exclude=0) == 1
    assert max_heap.peak(exclude=4) == 28


def test_peak_exclude_root_two_children_small(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=0)
    max_heap.push(thread_id=1, value=15)
    max_heap.push(thread_id=2, value=2)

    min_heap.push(thread_id=0, value=0)
    min_heap.push(thread_id=1, value=15)
    min_heap.push(thread_id=2, value=2)

    assert min_heap.peak(exclude=0) == 2
    assert max_heap.peak(exclude=1) == 2


def test_peak_exclude_root_one_child(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=0)
    max_heap.push(thread_id=1, value=15)

    min_heap.push(thread_id=0, value=0)
    min_heap.push(thread_id=1, value=15)

    assert min_heap.peak(exclude=0) == 15
    assert max_heap.peak(exclude=1) == 0


def test_peak_exclude_irrelevant(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=0)
    max_heap.push(thread_id=1, value=15)

    min_heap.push(thread_id=0, value=0)
    min_heap.push(thread_id=1, value=15)

    assert min_heap.peak(exclude=1) == 0
    assert max_heap.peak(exclude=0) == 15


def test_pop_empty(make_heap):
    max_heap, min_heap = make_heap
    with pytest.raises(RuntimeError):
        max_heap.pop()

    with pytest.raises(RuntimeError):
        min_heap.pop()
