"""Benchmark process_issue latency.

Copyright (c) 2025 Patrick Morrison. Licensed under the MIT License.

Usage:
  python -m scripts.benchmark --n 50 --mode safe
"""
from __future__ import annotations

import argparse
import time
from statistics import mean
from devops_agent.agent import UltimateAIAutonomousDevOps


def run(n: int, mode: str):
    logs = ["Out of memory error in service X" if i % 2 == 0 else "Connection timeout to db" for i in range(n)]
    latencies = []
    agent = UltimateAIAutonomousDevOps()
    import asyncio
    for log in logs:
        t0 = time.perf_counter()
        asyncio.run(agent.process_issue(log, mode))
        latencies.append((time.perf_counter() - t0) * 1000)
    agent.close()
    latencies.sort()
    p95 = latencies[int(0.95 * len(latencies)) - 1] if latencies else 0
    print(f"n={n} avg_ms={mean(latencies):.2f} p95_ms={p95:.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=20)
    p.add_argument('--mode', default='safe')
    args = p.parse_args()
    run(args.n, args.mode)


if __name__ == '__main__':
    main()
