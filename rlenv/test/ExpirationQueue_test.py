from TimeFeatures import ExpirationQueue
import pytest


@pytest.fixture
def queue():
    return ExpirationQueue(expiration=25)


def test_empty_queue(queue):
    assert queue.peak() == 0


def test_singleton_queue(queue):
    queue.push(15)
    assert queue.peak(time=16) == 1


def test_singleton_queue_expired(queue):
    queue.push(15)
    assert queue.peak(41) == 0


def test_several_none_expired(queue):
    queue.push(20)
    queue.push(24)
    queue.push(31)
    assert queue.peak(32) == 3


def test_several_one_expired(queue):
    queue.push(2)
    queue.push(24)
    queue.push(31)
    assert queue.peak(32) == 2


def test_several_multiple_expired(queue):
    queue.push(2)
    queue.push(9)
    queue.push(15)
    queue.push(24)
    queue.push(31)
    assert queue.peak(41) == 2


def test_several_all_expired(queue):
    queue.push(2)
    queue.push(9)
    queue.push(15)
    queue.push(24)
    queue.push(31)
    assert queue.peak(100) == 0


def test_adding_after_several_expired(queue):
    queue.push(2)
    queue.push(9)
    queue.push(15)
    queue.push(24)
    queue.push(31)
    assert queue.peak(39) == 3
    queue.push(40)
    queue.push(45)
    assert queue.peak(51) == 3


def test_adding_after_all_expired(queue):
    queue.push(2)
    queue.push(9)
    queue.push(15)
    queue.push(24)
    queue.push(31)
    assert queue.peak(39) == 3
    assert queue.peak(100) == 0

    queue.push(105)
    assert queue.peak(108) == 1
    queue.push(110)
    assert queue.peak(112) == 2


def test_adding_all_interleave(queue):
    queue.push(2)
    assert queue.peak(5) == 1
    queue.push(20)
    assert queue.peak(25) == 2
    queue.push(28)
    assert queue.peak(29) == 2
    queue.push(40)
    queue.push(45)
    assert queue.peak(50) == 3
    queue.push(60)
    assert queue.peak(64) == 3

    assert queue.peak(100) == 0

    queue.push(105)
    assert queue.peak(108) == 1
    queue.push(110)
    assert queue.peak(112) == 2