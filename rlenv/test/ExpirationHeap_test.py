from deprecated.TimeFeatures import ExpirationHeap
import pytest

@pytest.fixture
def make_heap():
    min_heap = ExpirationHeap(expiration=25, min_heap=True)
    max_heap = ExpirationHeap(expiration=25, min_heap=False)
    return max_heap, min_heap


def check_order(heap, expected, sizes, backups):
    for exp, size, backup in zip(expected, sizes, backups):
        a = heap.pop()
        assert a.value == exp
        assert heap.size == (size - 1)
        assert backup == (a.thread_id in heap.backups)


def test_push_simple(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=25, time=15)
    max_heap.push(thread_id=1, value=0, time=22)

    min_heap.push(thread_id=1, value=0, time=15)
    min_heap.push(thread_id=0, value=25, time=22)

    check_order(max_heap, [25, 0], [2, 1], [False, False])
    check_order(min_heap, [0, 25], [2, 1], [False, False])


def test_push_random_order(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=25, time=15)
    max_heap.push(thread_id=1, value=0, time=22)
    max_heap.push(thread_id=2, value=42, time=27)
    max_heap.push(thread_id=3, value=15, time=30)
    max_heap.push(thread_id=4, value=92, time=32)
    max_heap.push(thread_id=5, value=-4, time=33)

    min_heap.push(thread_id=0, value=25, time=15)
    min_heap.push(thread_id=1, value=0, time=22)
    min_heap.push(thread_id=2, value=42, time=27)
    min_heap.push(thread_id=3, value=15, time=30)
    min_heap.push(thread_id=4, value=92, time=32)
    min_heap.push(thread_id=5, value=-4, time=33)

    min_expected = sorted([25, 0, 42, 15, 92, -4])
    max_expected = min_expected[::-1]
    sizes = [6, 5, 4, 3, 2, 1]
    backups = [False, False, False, False, False, False]

    check_order(max_heap, max_expected, sizes, backups)
    check_order(min_heap, min_expected, sizes, backups)


def test_push_promote_multiple_backups(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=25, time=15)
    max_heap.push(thread_id=1, value=0, time=22)
    max_heap.push(thread_id=2, value=42, time=27)
    max_heap.push(thread_id=3, value=15, time=30)
    max_heap.push(thread_id=4, value=92, time=32)
    max_heap.push(thread_id=5, value=-4, time=33)
    assert not max_heap.promote(thread_id=4, value=90, time=37)
    assert not max_heap.promote(thread_id=4, value=89, time=38)

    max_expected = sorted([25, 0, 42, 15, 92, -4, 90, 89])[::-1]
    sizes = [7, 7, 6, 5, 4, 3, 2, 1]
    backups = [True, False, False, False, False, False, False, False]
    print(max_heap.backups[4])
    check_order(max_heap, max_expected, sizes, backups)


def test_push_promote_override_all_backups(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=25, time=15)
    max_heap.push(thread_id=1, value=0, time=22)
    max_heap.push(thread_id=2, value=42, time=27)
    max_heap.push(thread_id=3, value=15, time=30)
    max_heap.push(thread_id=4, value=92, time=32)
    max_heap.push(thread_id=5, value=-4, time=33)
    assert not max_heap.promote(thread_id=4, value=90, time=37)
    assert not max_heap.promote(thread_id=4, value=89, time=38)
    assert max_heap.promote(thread_id=4, value=93, time=39)
    max_expected = sorted([25, 0, 42, 15, 93, -4])[::-1]
    sizes = [6, 5, 4, 3, 2, 1]
    backups = [False, False, False, False, False, False]
    check_order(max_heap, max_expected, sizes, backups)


def test_peak_singleton_no_backups_not_expired(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=25, time=15)
    assert 25 == max_heap.peak(time=20)


def test_peak_singleton_no_backups_expired(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=25, time=15)
    assert 0 == max_heap.peak(time=51)


def test_peak_singleton_backups_no_backups_expired(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=25, time=15)
    max_heap.promote(thread_id=0, value=20, time=30)
    max_heap.promote(thread_id=0, value=18, time=35)
    print(max_heap.backups[0])
    assert 25 == max_heap.peak(time=17)
    assert 20 == max_heap.peak(time=51)
    assert 18 == max_heap.peak(time=56)
    assert 0 == max_heap.peak(time=80)
    max_heap.push(thread_id=0, value=2, time=72)
    max_heap.promote(thread_id=0, value=15, time=75)
    max_heap.promote(thread_id=0, value=17, time=80)
    assert 17 == max_heap.peak(time=85)
    assert 0 not in max_heap.backups


def test_peak_singleton_backups_multiple_backups_expired(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=25, time=15)
    max_heap.promote(thread_id=0, value=20, time=30)
    max_heap.promote(thread_id=0, value=18, time=35)
    max_heap.promote(thread_id=0, value=15, time=75)
    assert 15 == max_heap.peak(time=85)


def test_promote_override_one_backup_singleton(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=25, time=15)
    max_heap.promote(thread_id=0, value=20, time=30)
    max_heap.promote(thread_id=0, value=18, time=35)
    max_heap.promote(thread_id=0, value=19, time=40)
    max_heap.promote(thread_id=0, value=18.5, time=45)
    assert max_heap.peak(time=56) == 19
    assert max_heap.peak(time=66) == 18.5


def test_promote_override_multiple_backups_singleton(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=25, time=15)
    max_heap.promote(thread_id=0, value=20, time=30)
    max_heap.promote(thread_id=0, value=18, time=35)
    max_heap.promote(thread_id=0, value=22, time=40)
    max_heap.promote(thread_id=0, value=19, time=45)
    assert max_heap.peak(time=56) == 22
    assert max_heap.peak(time=66) == 19


def test_promote_override_all_backups_singleton(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=25, time=15)
    max_heap.promote(thread_id=0, value=20, time=20)
    max_heap.promote(thread_id=0, value=18, time=25)
    max_heap.promote(thread_id=0, value=22, time=30)
    assert max_heap.peak(time=31) == 25
    max_heap.promote(thread_id=0, value=24, time=35)
    assert max_heap.peak(time=41) == 24
    assert 0 not in max_heap.backups


def test_peak_no_backups_not_expired(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=25, time=15)
    max_heap.push(thread_id=1, value=35, time=30)
    max_heap.push(thread_id=2, value=30, time=35)
    assert max_heap.peak(time=40) == 35
    assert max_heap.peak(time=59) == 30

def test_peak_backups_no_backups_expired_still_top(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=25, time=15)
    max_heap.push(thread_id=1, value=35, time=30)
    max_heap.push(thread_id=2, value=30, time=35)
    max_heap.push(thread_id=3, value=29, time=36)
    max_heap.promote(thread_id=1, value=33, time=40)
    max_heap.promote(thread_id=1, value=31, time=45)
    assert max_heap.peak(time=56) == 33


def test_peak_backups_no_backups_expired_not_top(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=25, time=15)
    max_heap.push(thread_id=1, value=35, time=30)
    max_heap.push(thread_id=2, value=30, time=35)
    max_heap.push(thread_id=3, value=29, time=36)
    max_heap.promote(thread_id=1, value=22, time=40)
    max_heap.promote(thread_id=1, value=21, time=45)
    assert max_heap.peak(time=56) == 30

def test_peak_backups_one_backup_expired_still_top(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=25, time=15)
    max_heap.push(thread_id=1, value=35, time=30)
    max_heap.promote(thread_id=1, value=33, time=35)
    max_heap.promote(thread_id=1, value=31, time=45)
    max_heap.push(thread_id=2, value=30, time=50)
    max_heap.push(thread_id=3, value=29, time=55)
    assert max_heap.peak(time=61) == 31


def test_peak_backups_one_backup_expired_not_top(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=25, time=15)
    max_heap.push(thread_id=1, value=35, time=30)
    max_heap.promote(thread_id=1, value=31, time=35)
    max_heap.promote(thread_id=1, value=21, time=45)
    max_heap.push(thread_id=2, value=30, time=50)
    max_heap.push(thread_id=3, value=29, time=55)
    assert max_heap.peak(time=61) == 30


def test_peak_backups_multiple_backups_expired_still_top(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=25, time=15)

    max_heap.push(thread_id=1, value=45, time=30)
    max_heap.promote(thread_id=1, value=40, time=35)
    max_heap.promote(thread_id=1, value=33, time=40)
    max_heap.promote(thread_id=1, value=35, time=45)

    max_heap.push(thread_id=2, value=30, time=50)
    max_heap.promote(thread_id=2, value=29, time=55)
    max_heap.promote(thread_id=2, value=28, time=56)

    max_heap.push(thread_id=3, value=29, time=57)
    assert max_heap.peak(time=61) == 35

def test_peak_backups_multiple_backups_expired_not_top(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=25, time=15)

    max_heap.push(thread_id=1, value=45, time=30)
    max_heap.promote(thread_id=1, value=40, time=35)
    max_heap.promote(thread_id=1, value=26, time=40)
    max_heap.promote(thread_id=1, value=23, time=60)

    max_heap.push(thread_id=2, value=30, time=50)
    max_heap.promote(thread_id=2, value=29, time=55)
    max_heap.promote(thread_id=2, value=28, time=56)

    max_heap.push(thread_id=3, value=29, time=57)
    assert max_heap.peak(time=61) == 30
    assert max_heap.peak(time=84) == 23


def test_peak_backups_all_backups_expired(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=25, time=15)

    max_heap.push(thread_id=1, value=45, time=30)
    max_heap.promote(thread_id=1, value=40, time=35)
    max_heap.promote(thread_id=1, value=26, time=40)

    max_heap.push(thread_id=2, value=30, time=50)
    max_heap.promote(thread_id=2, value=29.5, time=55)
    max_heap.promote(thread_id=2, value=28, time=56)

    max_heap.push(thread_id=3, value=29, time=57)
    assert max_heap.peak(time=79) == 29.5


def test_peak_first_two_all_backups_expired(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=25, time=15)

    max_heap.push(thread_id=1, value=45, time=30)
    max_heap.promote(thread_id=1, value=40, time=35)
    max_heap.promote(thread_id=1, value=26, time=40)

    max_heap.push(thread_id=2, value=30, time=50)
    max_heap.promote(thread_id=2, value=29, time=55)
    max_heap.promote(thread_id=2, value=28, time=56)

    max_heap.push(thread_id=3, value=29, time=80)
    assert max_heap.peak(time=100) == 29

def test_peak_first_no_backups_expired_second_some_backups_expired_not_top(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=25, time=15)

    max_heap.push(thread_id=1, value=45, time=20)
    max_heap.promote(thread_id=1, value=27, time=50)
    max_heap.promote(thread_id=1, value=26, time=55)

    max_heap.push(thread_id=2, value=30, time=20)
    max_heap.promote(thread_id=2, value=29, time=25)
    max_heap.promote(thread_id=2, value=26, time=35)

    assert max_heap.peak(time=56) == 27


def test_peak_all_entries_all_backups_expired(make_heap):
    max_heap, min_heap = make_heap
    max_heap.push(thread_id=0, value=25, time=15)

    max_heap.push(thread_id=1, value=45, time=20)
    max_heap.promote(thread_id=1, value=27, time=50)
    max_heap.promote(thread_id=1, value=26, time=55)

    max_heap.push(thread_id=2, value=30, time=20)
    max_heap.promote(thread_id=2, value=29, time=25)
    max_heap.promote(thread_id=2, value=26, time=35)

    assert max_heap.peak(time=100) == 0