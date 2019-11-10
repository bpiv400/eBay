from time.TimeFeatures import FeatureCounter
import pytest


@pytest.fixture
def counter():
    return FeatureCounter('feature')


def check_contents(counter, exp):
    contents = counter.contents
    for key, val in exp.items():
        assert val == contents[key]
    assert len(contents) == len(exp)


def test_empty(counter):
    assert counter.peek() == 0
    assert counter.peek(exclude=1) == 0
    check_contents(counter, {})


def test_push_singleton(counter):
    counter.increment(thread_id=1)
    assert counter.peek() == 1
    assert counter.peek(exclude=1) == 0
    check_contents(counter, {
        1: 1
    })


def test_increment_singleton(counter):
    counter.increment(thread_id=1)
    counter.increment(thread_id=1)
    assert counter.peek() == 2
    assert counter.peek(exclude=1) == 0
    check_contents(counter, {
        1: 2
    })


def test_increment_singleton_twice(counter):
    counter.increment(thread_id=1)
    counter.increment(thread_id=1)
    counter.increment(thread_id=1)
    assert counter.peek() == 3
    assert counter.peek(exclude=1) == 0
    check_contents(counter, {
        1: 3
    })


def test_remove_singleton(counter):
    counter.increment(thread_id=1)
    counter.increment(thread_id=1)
    counter.remove(thread_id=1)
    assert counter.peek() == 0
    assert counter.peek(exclude=1) == 0
    check_contents(counter, {})


def test_increment_after_remove_singleton(counter):
    counter.increment(thread_id=1)
    counter.increment(thread_id=1)
    counter.remove(thread_id=1)
    counter.increment(thread_id=1)
    assert counter.peek() == 1
    assert counter.peek(exclude=1) == 0
    check_contents(counter, {
        1: 1
    })


def test_remove_not_found(counter):
    counter.increment(thread_id=1)
    counter.increment(thread_id=2)
    counter.remove(thread_id=3)
    assert counter.peek() == 2
    assert counter.peek(exclude=1) == 1
    check_contents(counter, {
        1: 1,
        2: 1
    })


def test_remove_none(counter):
    counter.increment(thread_id=1)
    counter.increment(thread_id=2)
    counter.remove(thread_id=None)
    assert counter.peek() == 2
    assert counter.peek(exclude=1) == 1
    check_contents(counter, {
        1: 1,
        2: 1
    })


def test_increment_none(counter):
    counter.increment(thread_id=1)
    counter.increment(thread_id=2)
    counter.increment(thread_id=None)
    assert counter.peek() == 2
    assert counter.peek(exclude=1) == 1
    check_contents(counter, {
        1: 1,
        2: 1
    })


def test_push_multiple(counter):
    counter.increment(thread_id=1)
    counter.increment(thread_id=2)
    counter.increment(thread_id=3)
    assert counter.peek() == 3
    assert counter.peek(exclude=1) == 2
    assert counter.peek(exclude=2) == 2
    assert counter.peek(exclude=3) == 2
    check_contents(counter, {
        1: 1,
        2: 1,
        3: 1
    })


def test_increment_several(counter):
    counter.increment(thread_id=1)
    counter.increment(thread_id=1)
    counter.increment(thread_id=2)
    counter.increment(thread_id=3)
    counter.increment(thread_id=3)
    counter.increment(thread_id=1)
    assert counter.peek() == 6
    assert counter.peek(exclude=1) == 3
    assert counter.peek(exclude=2) == 5
    assert counter.peek(exclude=3) == 4
    check_contents(counter, {
        1: 3,
        2: 1,
        3: 2
    })


def test_increment_weave(counter):
    counter.increment(thread_id=1)
    counter.increment(thread_id=1)
    counter.increment(thread_id=2)
    counter.increment(thread_id=1)
    counter.increment(thread_id=3)
    counter.increment(thread_id=3)
    counter.increment(thread_id=1)
    counter.increment(thread_id=2)
    counter.increment(thread_id=3)
    assert counter.peek() == 9
    assert counter.peek(exclude=1) == 5
    assert counter.peek(exclude=2) == 7
    assert counter.peek(exclude=3) == 6
    check_contents(counter, {
        1: 4,
        2: 2,
        3: 3
    })


def test_increment_remove_weave(counter):
    counter.increment(thread_id=1)
    counter.increment(thread_id=1)
    counter.increment(thread_id=2)
    counter.remove(thread_id=1)
    counter.increment(thread_id=3)
    counter.increment(thread_id=3)
    counter.increment(thread_id=1)
    counter.increment(thread_id=2)
    counter.remove(thread_id=3)
    counter.increment(thread_id=2)
    counter.increment(thread_id=1)
    assert counter.peek() == 5
    assert counter.peek(exclude=1) == 3
    assert counter.peek(exclude=2) == 2
    assert counter.peek(exclude=3) == 5
    check_contents(counter, {
        1: 2,
        2: 3
    })


def test_remove_all(counter):
    counter.increment(thread_id=1)
    counter.increment(thread_id=1)
    counter.increment(thread_id=2)
    counter.increment(thread_id=1)
    counter.remove(thread_id=1)
    counter.remove(thread_id=2)
    counter.increment(thread_id=3)
    counter.increment(thread_id=3)
    counter.increment(thread_id=1)
    counter.remove(thread_id=1)
    counter.increment(thread_id=2)
    counter.increment(thread_id=3)
    counter.remove(thread_id=3)
    counter.remove(thread_id=2)
    assert counter.peek() == 0
    assert counter.peek(exclude=1) == 0
    assert counter.peek(exclude=2) == 0
    assert counter.peek(exclude=3) == 0
    check_contents(counter, {})
