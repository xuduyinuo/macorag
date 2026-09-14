"""Coalesce concurrent, synchronous evaluation calls without changing their seeds."""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future


class GenerateBatcher:
    def __init__(self, send, *, max_size=8, wait_ms=20):
        if max_size < 1 or wait_ms < 0:
            raise ValueError("Batch size must be positive and batch wait nonnegative")
        self.send = send
        self.max_size = max_size
        self.wait_seconds = wait_ms / 1000
        self.condition = threading.Condition()
        self.pending = []
        self.busy = False
        self.batches = 0
        self.prompts = 0
        self.largest_batch = 0

    def submit(self, endpoint, payload):
        if payload.get("n", 1) != 1 or len(payload["prompts"]) != 1 or len(payload["seeds"]) != 1:
            raise ValueError("Evaluation batching requires one prompt, one seed and n=1")
        parameters = {k: v for k, v in payload.items() if k not in ("prompts", "seeds")}
        key = (endpoint, json.dumps(parameters, sort_keys=True))
        future = Future()
        with self.condition:
            self.pending.append((key, payload, future))
            self.condition.notify_all()
        while True:
            with self.condition:
                if future.done():
                    return future.result()
                if self.busy:
                    self.condition.wait()
                    continue
                self.busy = True
            # One caller acts as the dispatcher. Other callers enqueue while it
            # waits or performs HTTP I/O; no persistent background thread exists.
            batch = []
            try:
                time.sleep(self.wait_seconds)
                with self.condition:
                    first_key = self.pending[0][0]
                    remaining = []
                    for item in self.pending:
                        if item[0] == first_key and len(batch) < self.max_size:
                            batch.append(item)
                        else:
                            remaining.append(item)
                    self.pending = remaining
                merged = dict(batch[0][1])
                merged["prompts"] = [item[1]["prompts"][0] for item in batch]
                merged["seeds"] = [item[1]["seeds"][0] for item in batch]
                response = self.send(first_key[0], merged)
                completions = response["completion_ids"]
                if len(completions) != len(batch):
                    raise ValueError("Batched generation returned the wrong number of completions")
                for index, (_, _, result) in enumerate(batch):
                    result.set_result({"completion_ids": [completions[index]]})
                with self.condition:
                    self.batches += 1
                    self.prompts += len(batch)
                    self.largest_batch = max(self.largest_batch, len(batch))
            except BaseException as exc:
                for _, _, result in batch:
                    if not result.done():
                        result.set_exception(exc)
                if not isinstance(exc, Exception):
                    raise
            finally:
                with self.condition:
                    self.busy = False
                    self.condition.notify_all()

    def statistics(self):
        with self.condition:
            return {"batches": self.batches, "prompts": self.prompts,
                    "mean_batch_size": self.prompts / self.batches if self.batches else 0,
                    "largest_batch": self.largest_batch}
