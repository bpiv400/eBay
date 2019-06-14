import heapq
from Environment import Environment


def test_random():
    assert 1 == 1


def test_env_setup():
    env = Environment(2)
    slrs = env.slrs
    for i in range(10):
        env.initialize_slr(slrs[i])
    assert (len(env.slr_queues) == 10)
    for i in range(10):
        curr = -10
        for _ in range(len(env.slr_queues[i])):
            event = heapq.heappop(env.slr_queues[i])
            prev = curr
            curr = event.priority
            assert (prev <= curr)
