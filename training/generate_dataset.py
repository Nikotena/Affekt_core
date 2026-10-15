#!/usr/bin/env python3
"""
Fine-tuning dataset pipeline (teacher: TBrilliance by default).

    python training/generate_dataset.py generate --n 200 --out data/raw.jsonl
    python training/generate_dataset.py filter --raw data/raw.jsonl --out-dir data
    python training/generate_dataset.py review --out-dir data      # writes data/review.md

`generate` is resumable (it skips scenario ids already in the output), so it can run in
chunks on a cloud GPU. `filter` writes zero-shot chat samples in the same prompt format
the engine uses at inference, split into train / eval_seen / eval_unseen.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "test"))

from modules.LLMLayer import build_messages, narrate_action, set_example_seed  # noqa: E402
from training.filters import drop_near_duplicates, normalize_prices, reject_reason  # noqa: E402
from training.scenarios import Scenario, generate_scenarios, make_payload  # noqa: E402

DEFAULT_TEACHER = os.path.join(ROOT, "model", "LFM2.5-2.6B-Q3.8-TBrilliance-NEO-MAX-IQ4_XS.gguf")
BENCH_ROLES = {"NPC_0": "Bartender", "NPC_1": "Merchant", "NPC_2": "Guard", "NPC_3": "Healer",
               "NPC_4": "Blacksmith"}
KIND_BY_NARRATION = {
    "The player attacks you.": "attack", "The player greets you.": "greet",
    "The player offers to help you.": "help", "The player wants to buy something from you.": "buy",
    "The player wants to trade with you.": "trade", "The player asks you a question.": "ask",
}


def bench_keys():
    """Scenario keys of test/eval_llm.py: they must never appear in training data."""
    from eval_llm import SCENARIOS
    keys = set()
    for npc_id, mood, action, _expected in SCENARIOS:
        kind = KIND_BY_NARRATION.get(narrate_action(action), "other")
        role = BENCH_ROLES[npc_id]
        if kind == "buy":
            keys.add((role, mood, kind, "Potion" if "potion" in action else "Sword"))
        else:
            keys.add((role, mood, kind))
    return keys


def read_jsonl(path):
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path, row, retries=5):
    """Append one row, opening and closing the file each time (long runs hit sporadic
    Windows I/O errors on a long-lived handle; the row is retried instead of lost)."""
    line = json.dumps(row, ensure_ascii=False) + "\n"
    for attempt in range(retries):
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
            return
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(1 + attempt)


def make_teacher(args):
    """Build the teacher backend. Returns (backend, label recorded in every row)."""
    if args.teacher in ("openrouter", "amd"):
        from modules.ApiBackend import AMD_DEFAULT_MODEL, AMDBackend, OPENROUTER_DEFAULT_MODEL, OpenRouterBackend
        if not args.teacher_license:
            sys.exit(f"--teacher-license is required with --teacher {args.teacher}: write down the "
                     "license/terms that allow training on this model's outputs (check its model card).")
        cls, default = (AMDBackend, AMD_DEFAULT_MODEL) if args.teacher == "amd" else (OpenRouterBackend, OPENROUTER_DEFAULT_MODEL)
        model = args.model or default
        kw = {k: v for k, v in (("min_interval", args.min_interval), ("timeout", getattr(args, "timeout", None)))
              if v is not None}
        return cls(model, temperature=args.temperature, **kw), model
    from modules.LLMLayer import LlamaCppBackend
    model = args.model or DEFAULT_TEACHER
    return LlamaCppBackend(model, temperature=args.temperature, n_ctx=1024), os.path.basename(model)


def cmd_generate(args):
    scenarios = generate_scenarios(args.n, seed=args.seed, exclude_keys=bench_keys())
    done = {r["id"] for r in read_jsonl(args.out)}
    todo = [s for s in scenarios if s.id not in done]
    print(f"{len(scenarios)} scenarios, {len(done)} already generated, {len(todo)} to do")
    if not todo:
        return
    teacher, label = make_teacher(args)
    teacher.shots = args.shots
    set_example_seed(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    start, failed, done_n = time.time(), 0, 0

    def work(sc):
        payload = make_payload(sc)
        return sc, payload, teacher.generate(payload, sc.action)

    # Responses are slow (tens of seconds), so several run at once; the backend spaces their
    # start times so the API's per-minute limit still holds. Only this thread writes the file.
    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    futures = {pool.submit(work, sc): sc for sc in todo}
    try:
        for fut in as_completed(futures):
            sc = futures[fut]
            done_n += 1
            try:
                _, payload, reply = fut.result()
            except Exception as e:  # keep going on transient errors; the row is retried on the next run
                if any(code in str(e) for code in ("HTTP 401", "HTTP 402", "HTTP 403")):
                    sys.exit(f"stopping: {e}")
                failed += 1
                print(f"  skipped {sc.id}: {str(e)[:120]}", flush=True)
                if failed >= args.max_failures:
                    sys.exit(f"stopping after {failed} failures")
                continue
            row = {**sc.to_dict(), "payload": payload, "reply": reply, "teacher": label}
            if args.teacher in ("openrouter", "amd"):
                row["teacher_license"] = args.teacher_license
            append_jsonl(args.out, row)
            if done_n % 10 == 0 or done_n == len(todo):
                rate = (time.time() - start) / done_n
                print(f"  {done_n}/{len(todo)}  {rate:.1f}s each, ~{rate * (len(todo) - done_n) / 60:.0f} min left", flush=True)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def to_sample(row):
    """Zero-shot chat sample: the exact prompt the fine-tuned model sees at inference."""
    messages = build_messages(row["payload"], row["action"], mood=row["mood"], no_think=True, shots=0)
    messages.append({"role": "assistant", "content": row["reply"]})
    return {"id": row["id"], "mood": row["mood"], "kind": row["kind"], "split": row["split"],
            "messages": messages}


def cmd_filter(args):
    paths = [args.raw] if isinstance(args.raw, str) else args.raw
    raw = [row for path in paths for row in read_jsonl(path)]
    reasons, clean = Counter(), []
    for row in raw:
        fixed = normalize_prices(row["reply"])  # "twelve coins" -> "12 coins"
        if fixed != row["reply"]:
            row["reply_raw"], row["reply"] = row["reply"], fixed
        why = reject_reason(row["reply"], row["mood"], row["expected"], name=row["name"],
                            role=row["role"], allow_profanity=getattr(args, "allow_profanity", False),
                            place=row["place"], price=row["price"],
                            price_shown=any("coins." in l for l in row["payload"]))
        if why:
            reasons[why] += 1
            row["rejected"] = why
        else:
            clean.append(row)
    kept = []
    for split in ("train", "eval_seen", "eval_unseen"):
        part = drop_near_duplicates([r for r in clean if r["split"] == split])
        reasons["near_duplicate"] += len([r for r in clean if r["split"] == split]) - len(part)
        kept += part
        write_jsonl(os.path.join(args.out_dir, f"{split}.jsonl"), [to_sample(r) for r in part])
    write_jsonl(os.path.join(args.out_dir, "rejected.jsonl"), [r for r in raw if r.get("rejected")])
    write_jsonl(os.path.join(args.out_dir, "kept_rows.jsonl"), kept)

    print(f"raw {len(raw)}  ->  kept {len(kept)} ({100 * len(kept) / max(len(raw), 1):.0f}%)")
    print("rejected by reason:", dict(reasons.most_common()))
    print("kept by split:", dict(Counter(r["split"] for r in kept)))
    print("kept by mood: ", dict(Counter(r["mood"] for r in kept)))
    print("kept by kind: ", dict(Counter(r["kind"] for r in kept)))


def cmd_review(args):
    rows = read_jsonl(os.path.join(args.out_dir, "kept_rows.jsonl"))
    rejected = read_jsonl(os.path.join(args.out_dir, "rejected.jsonl"))
    rng = random.Random(args.seed)
    lines = ["# Manual review sample", "",
             "Mark each line OK / BAD and note why. Target: >= 80 % OK before scaling up.", "",
             "## Kept (what the model would learn from)", ""]
    for r in rng.sample(rows, min(args.k, len(rows))):
        lines.append(f"- [ ] **{r['role']} / {r['mood']} / {r['action']}**"
                     + (f" ({r['item']} {r['price']})" if r["expected"] else "")
                     + f"\n  > {r['reply']}")
    lines += ["", "## Rejected (check the filter is not throwing away good lines)", ""]
    for r in rng.sample(rejected, min(10, len(rejected))):
        lines.append(f"- `{r['rejected']}` **{r['role']} / {r['mood']} / {r['action']}**\n  > {r['reply']}")
    path = os.path.join(args.out_dir, "review.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--n", type=int, default=200)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default="data/raw.jsonl")
    g.add_argument("--teacher", choices=["local", "openrouter", "amd"], default="local")
    g.add_argument("--model", default=None, help="GGUF path (local) or API model id (openrouter / amd)")
    g.add_argument("--teacher-license", default="", help="license/terms allowing training on the outputs (required for openrouter / amd)")
    g.add_argument("--min-interval", type=float, default=None, help="seconds between API calls (default: 3.0 openrouter, 3.2 amd)")
    g.add_argument("--workers", type=int, default=1,
                   help="parallel API calls (amd: 4 keeps under ~20 requests/min with ~40 s replies)")
    g.add_argument("--timeout", type=float, default=None, help="seconds to wait for one API reply (default 120)")
    g.add_argument("--max-failures", type=int, default=20)
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--shots", type=int, default=3)
    g.set_defaults(fn=cmd_generate)

    f = sub.add_parser("filter")
    f.add_argument("--raw", nargs="+", default=["data/raw.jsonl"], help="one or more raw files to merge")
    f.add_argument("--out-dir", default="data")
    f.add_argument("--allow-profanity", action="store_true")
    f.set_defaults(fn=cmd_filter)

    r = sub.add_parser("review")
    r.add_argument("--out-dir", default="data")
    r.add_argument("--k", type=int, default=30)
    r.add_argument("--seed", type=int, default=1)
    r.set_defaults(fn=cmd_review)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
