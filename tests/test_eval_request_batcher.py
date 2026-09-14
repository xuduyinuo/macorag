from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from evaluation.request_batcher import GenerateBatcher


def test_batch_preserves_seeds_order_and_endpoint():
    calls = []
    barrier = threading.Barrier(8)

    def send(endpoint, payload):
        calls.append((endpoint, payload))
        return {"completion_ids": [[seed] for seed in payload["seeds"]]}

    batcher = GenerateBatcher(send, max_size=4, wait_ms=30)

    def run(i):
        barrier.wait()
        return batcher.submit(str(i % 2), {"prompts": [str(i)], "seeds": [i], "n": 1})

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, range(8)))
    assert results == [{"completion_ids": [[i]]} for i in range(8)]
    for endpoint, payload in calls:
        assert len(payload["prompts"]) <= 4
        assert payload["prompts"] == [str(seed) for seed in payload["seeds"]]
        assert all(str(seed % 2) == endpoint for seed in payload["seeds"])
    assert batcher.statistics()["prompts"] == 8
    assert batcher.statistics()["largest_batch"] > 1


@pytest.mark.parametrize("wrong_count", [False, True])
def test_batch_errors_fan_out_and_next_call_recovers(wrong_count):
    failing = True

    def send(endpoint, payload):
        if failing:
            if wrong_count:
                return {"completion_ids": []}
            raise RuntimeError("unavailable")
        return {"completion_ids": [[x] for x in payload["seeds"]]}

    batcher = GenerateBatcher(send, max_size=4)
    payload = {"prompts": ["q"], "seeds": [7], "n": 1}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(batcher.submit, "endpoint", payload) for _ in range(4)]
        for future in futures:
            with pytest.raises((RuntimeError, ValueError)):
                future.result(timeout=2)
    failing = False
    assert batcher.submit("endpoint", payload) == {"completion_ids": [[7]]}


def test_sampling_parameters_are_not_mixed():
    def send(endpoint, payload):
        return {"completion_ids": [[payload["temperature"]] for _ in payload["prompts"]]}

    batcher = GenerateBatcher(send)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(batcher.submit, "url", {
            "prompts": ["q"], "seeds": [i], "temperature": i,
        }) for i in range(4)]
        assert [f.result(timeout=2) for f in futures] == [{"completion_ids": [[i]]} for i in range(4)]
