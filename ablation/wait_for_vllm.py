#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--kind", choices=("openai", "training"), default="openai")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    endpoint = args.base_url.rstrip("/") + ("/models" if args.kind == "openai" else "/health/")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + args.timeout
    last_error = "server not ready"
    while time.monotonic() < deadline:
        try:
            os.kill(args.pid, 0)
        except ProcessLookupError as exc:
            raise SystemExit(f"vLLM launcher exited before readiness: pid={args.pid}") from exc
        try:
            with opener.open(endpoint, timeout=5.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if args.kind == "openai":
                model_ids = {str(item.get("id") or "") for item in payload.get("data", [])}
                ready = args.model in model_ids
                last_error = f"served models do not include {args.model!r}: {sorted(model_ids)}"
            else:
                ready = payload.get("status") == "ok" and str(payload.get("model") or "") == args.model
                last_error = f"training health payload mismatch: {payload}"
            if ready:
                print(f"vLLM ready: model={args.model} endpoint={endpoint}")
                return
        except Exception as exc:  # readiness polling intentionally retries transport and JSON errors
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2.0)
    raise SystemExit(f"Timed out waiting for vLLM after {args.timeout}s: {last_error}")


if __name__ == "__main__":
    main()
