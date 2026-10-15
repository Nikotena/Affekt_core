#!/usr/bin/env python3
"""
Latency, error-rate and fact-accuracy benchmark for the hosted model (AMD Token Factory).

    python test/bench_api.py --calls 30 --workers 1 --label sequential
    python test/bench_api.py --calls 30 --workers 4 --label concurrent4 --out docs/benchmarks/concurrent4.json

Each call is a real engine prompt: the scenarios of test/eval_llm.py (NPC x mood x action) built
with AffektEngine.build_payload, sent with the same settings create_engine() uses for the API
(2 few-shot examples, 5 example sets, temperature 0.5, 30 s timeout, 2 retries).

Reported:
  end-to-end   wall time of one engine call: waiting for a request slot + network + retries
  network      time of each HTTP attempt only (what the server took)
  errors       calls that raised (the engine would have used a canned line)
  retried      calls that needed at least one retry (HTTP 429/5xx or timeout)
  checks       the eval_llm checks on each reply (facts = the price the engine gave is in the reply)
  throughput   completed calls per minute (the team key allows about 20)
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "test"))

from AffektEngine import AffektEngine  # noqa: E402
from eval_llm import MOOD_VECTORS, SCENARIOS, run_checks  # noqa: E402
from modules.ApiBackend import AMDBackend  # noqa: E402
from modules.LLMLayer import set_example_seed, set_example_variants  # noqa: E402


def percentile(values, q):
    """Nearest-rank percentile (q in 0-100); no interpolation, so it is easy to verify by hand."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * q // 100))  # ceil
    return ordered[int(rank) - 1]


class TimedOpener:
    """Wraps urlopen to time every HTTP attempt and count failures (thread-safe)."""

    def __init__(self):
        self.attempts, self.failures = [], 0
        self._lock = threading.Lock()

    def __call__(self, request, timeout=None):
        start = time.time()
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except Exception:
            with self._lock:
                self.failures += 1
                self.attempts.append(time.time() - start)
            raise
        with self._lock:
            self.attempts.append(time.time() - start)
        return response


def summarize(values):
    return {"median_s": round(statistics.median(values), 2) if values else None,
            "p95_s": round(percentile(values, 95), 2) if values else None,
            "mean_s": round(statistics.mean(values), 2) if values else None,
            "max_s": round(max(values), 2) if values else None, "n": len(values)}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--calls", type=int, default=30)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--label", default="run")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", help="write the full result as JSON")
    args = ap.parse_args()

    set_example_variants(5)
    set_example_seed(args.seed)
    engine = AffektEngine()
    engine.initialize_modules()
    jobs = []  # prompts are built up front, in one thread (the engine is not thread-safe)
    for i in range(args.calls):
        npc_id, mood, action, expected = SCENARIOS[i % len(SCENARIOS)]
        engine.personality_for(npc_id).vector = list(MOOD_VECTORS[mood])
        payload, _ = engine.build_payload(npc_id, action)
        jobs.append((npc_id, mood, action, expected, payload))

    opener = TimedOpener()
    backend = AMDBackend(temperature=0.5, timeout=30.0, retries=2, opener=opener)
    backend.shots = 2

    def one(job):
        npc_id, mood, action, expected, payload = job
        start = time.time()
        try:
            reply, error = backend.generate(payload, action), None
        except Exception as e:  # noqa: BLE001 - any failure means "canned line" in the engine
            reply, error = "", str(e)[:120]
        elapsed = time.time() - start
        return {"npc": npc_id, "mood": mood, "action": action, "seconds": elapsed, "error": error,
                "reply": reply, "checks": run_checks(reply, mood, expected) if reply else {}}

    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(one, jobs))
    wall = time.time() - started

    ok = [r for r in results if not r["error"]]
    check_names = sorted({name for r in ok for name in r["checks"]})
    checks = {name: [sum(r["checks"].get(name, True) for r in ok), len(ok)] for name in check_names}
    # a call "retried" if the backend made more HTTP attempts than calls (counted globally)
    extra_attempts = max(0, len(opener.attempts) - len(jobs))
    summary = {
        "label": args.label, "model": backend.model, "calls": len(jobs), "workers": args.workers,
        "wall_seconds": round(wall, 1), "throughput_per_min": round(60 * len(ok) / wall, 1),
        "errors": len(results) - len(ok), "error_rate": round((len(results) - len(ok)) / len(results), 3),
        "http_attempts": len(opener.attempts), "extra_attempts_from_retries": extra_attempts,
        "end_to_end": summarize([r["seconds"] for r in ok]),
        "network_per_attempt": summarize(opener.attempts),
        "checks_passed": checks,
        "error_samples": sorted({r["error"] for r in results if r["error"]})[:3],
    }

    print(f"[{args.label}] {len(jobs)} calls, {args.workers} worker(s), {wall:.0f}s wall, "
          f"{summary['throughput_per_min']} calls/min")
    for name in ("end_to_end", "network_per_attempt"):
        s = summary[name]
        print(f"  {name:20s} median {s['median_s']}s  p95 {s['p95_s']}s  max {s['max_s']}s  (n={s['n']})")
    print(f"  errors {summary['errors']}/{len(jobs)} ({100 * summary['error_rate']:.1f}%), "
          f"extra HTTP attempts from retries: {extra_attempts}")
    print("  checks " + "  ".join(f"{k} {p}/{n}" for k, (p, n) in checks.items()))
    for e in summary["error_samples"]:
        print("  error sample:", e)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({**summary, "results": [{k: v for k, v in r.items() if k != "reply"} | {"reply": r["reply"]}
                                              for r in results]}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
