# Evaluation: what was measured, how, and what is still missing

**Model note.** All model measurements below were taken with Gemma-4-31B-IT on AMD Token Factory. Gemma has since
been removed from that catalog and the default is now MiMo-V2.6-Flash, which was only spot-checked (see
[amd.md](amd.md)); re-run the commands to measure it.

Everything here can be re-run with the commands in each section. Numbers were measured on
2026-10-14 against the shared free tier of AMD Token Factory, so latency in particular depends on
the load at that moment. Raw results are in [`docs/benchmarks/`](benchmarks/).

## Status at a glance

| Area | Implemented | Measured | Planned |
|---|---|---|---|
| Deterministic core (emotion, memory, graph, triggers) | Yes | 194 unit tests pass | |
| Per-NPC temperament | Yes | Behaviour tests (`test_temperament.py`) | Tuning with player feedback |
| Intent parser (rules) | Yes | 95% on a 179-line bank; 71% / 54% on own 150 / 50 labelled lines | More rules or a trained classifier |
| LLM intent fallback | Yes, off by default | 54% to 64% on 50 held-out lines (not statistically conclusive, see below) | A larger held-out set |
| NPC dialogue with Gemma-4-31B-IT | Yes | Latency, errors and 6 automatic checks on 60 calls | Latency benchmark of the hardware itself |
| Price check at runtime (`FactGuard`) | Yes | 0 false alarms on 167 Gemma replies; caught 1 real error in 299 replies from other models | Check items and NPC state, not only prices |
| Concurrency | Rate limiting and retries only | 1 vs 4 workers on 30 calls each | Async request queue, response cache |
| Fine-tuned local model | No | n/a | See [finetune_compact.md](finetune_compact.md) |

## 1. Test suite

```powershell
python -m unittest discover -s test        # 194 tests, about half a second, no network, no model
```

They cover the engine wiring, personality maths, memory, graph, intent parser, name handling,
prompt rules (for example that a price question makes the reply state the price), temperaments,
API client behaviour with a simulated HTTP layer (retries, 429 handling, key never printed),
fallback when the API fails, the runtime price guard, and the benchmark statistics helpers.

### Core module audit

The core modules were also audited with an adversarial script (64 checks: boundaries, invalid
input, fuzzing, determinism across processes, a simulated Neo4j driver) in addition to the unit
tests. It found 8 real problems, all fixed and now covered by `test/test_core_modules.py`:

| Module | Problem found | Fix |
|---|---|---|
| MemorySet | One 20-entry buffer shared by all NPCs: 30 lines to Marta erased all 5 of Kael's | One buffer per NPC |
| MemorySet | `importance_of([])` raised an error | Returns 0.0 |
| ContextFunction | Triggers matched substrings: "I forgive you" fired `give`, "a counsellor" fired `sell`, "buyer" fired `buy` | Whole words only (with forms like selling, gave, bought) |
| GraphDatabase | Forgotten memories were removed in memory but stayed in Neo4j and came back on load | Removed nodes are deleted on the next sync |
| GraphDatabase | NaN or infinity in a property produced invalid Cypher | Written as `null` |
| GraphDatabase | One node with an unknown label, or an edge to a skipped node, aborted `load_from_neo4j` | Skipped with a message |

Checked and fine: the 4D vector stays in range over 15,000 random actions for every temperament,
the input delta is never modified, NPCs do not share state, emotions fade or persist according to
inertia, the payload is identical across processes (6 different hash seeds), and `remove_node`
leaves no dangling edge. The engine handles empty, very long and unicode input and an NPC that is
not in the graph.

**Known limits, not fixed.**
- Long-term memory has no size cap or forgetting: 2,000 consolidated events keep 2,000 concepts.
  Fine for a session of hours, not for a persistent world.
- Neo4j was only tested against a simulated driver, not a real server.

## 2. Intent parser evaluation

**Data.** 200 player lines written by hand (`text.txt`) and labelled with one of 14 intents by the
author. The labelled lines are in `test/data/labels.json`. They were split once, at random with
seed 42, into 150 **tuning** lines (`labels_dev.json`) used to write and adjust the rules, and 50
**held-out** lines (`labels_reserve.json`) that were not read while tuning. The evaluation script
never prints the held-out text.

**Metric.** Exact match of the predicted intent against the label (14 classes) on lines that are not
balanced: 60 `complain`, 54 `other`, 37 `ask`, and 49 spread over the other classes (`haggle` has none).

**How the 54% to 64% was computed.** On the 50 held-out lines the rules alone got 27 right (54%).
For every line the rules answered `other` (26 lines), Gemma-4-31B-IT was asked for one label at
temperature 0; with that, 32 were right (64%).

| Held-out set (50 lines) | Result |
|---|---|
| Rules only | 27/50 = 54%, 95% interval 40-67% |
| Rules + Gemma-4-31B-IT | 32/50 = 64%, 95% interval 50-76% |
| Lines the model fixed / broke | 9 fixed, 4 broke (net +5) |

**Reading it honestly.** The two intervals overlap widely and a sign test on the 13 lines that
changed gives p = 0.27, so with 50 lines the gain is suggestive but **not statistically
conclusive**. The direction was the same on the 150 tuning lines (71% to 77%), but those lines
were also used to tune the rules, so they are not independent evidence. The labels come from a
single person, and `complain` is a broad category, so some disagreement with other annotators is
expected.

```powershell
python test\eval_labeled.py                  # tuning lines, rules only
python test\eval_labeled.py --reserve --llm  # held-out lines with the model (about 26 API calls)
```

An older bank of 179 lines (`python test\eval_intent.py --set all`) scores 95% for the rules; it
was written earlier and the rules were developed against it, so it overstates real-world accuracy.

## 3. Do the replies respect the game facts?

The engine decides the facts and the model only phrases them. Three layers check this.

1. **Prompt.** Facts come only from the world graph (role, place, price, item). For purchases,
   haggling and price questions the prompt ends with an explicit instruction such as
   `Say the price: 25 coins.` The prompt is also checked by unit tests.
2. **Evaluation checks** (`test/eval_llm.py`, `run_checks`). For each reply: not empty, at most 3
   sentences and 220 characters, no meta-language ("the player", "mood"), no copying of the
   few-shot examples, and `facts`: the engine's price appears in the reply (spelled-out numbers
   are converted to digits first).
3. **Dataset filter** (`training/filters.py`). Rejects replies that mention a price when none was
   given, a price different from the engine's, or another location's name.
4. **Runtime guard** (`modules/FactGuard.py`, used by `AffektEngine`). After the model answers,
   every amount of coins in the reply (spelled-out numbers converted to digits) is compared with
   the prices the engine put in the prompt. A number the player just said is allowed, so the NPC
   can answer a counter-offer. If the reply states a wrong price for something the NPC sells it
   is replaced with a plain line using the engine's price ("The Healing potion costs 25 coins.");
   if it states a price when the NPC sells nothing it is replaced by the canned line for the mood.
   The result is returned as `price_guard` (`ok`, `corrected`, `blocked` or `not_checked`).

**Results** on 60 real calls (30 sequential and 30 concurrent, below):

| Check | Sequential | 4 workers |
|---|---|---|
| Price stated correctly (facts) | 29/30 | 30/30 |
| Short, no meta-language, no copying | 30/30 each | 30/30 each |
| Tone (hostile/friendly keywords) | 29/30 | 28/30 |

The one `facts` failure is a false alarm of the checker: the model wrote "twenty-five" without
the word "coins", which the normalizer does not convert; the price was correct. The three `tone`
failures are clearly hostile replies ("I'll wipe your blood off my floor") that the keyword
heuristic does not recognize, so the tone check is only a rough guide.

**Runtime guard on saved replies** (`python test\check_guard.py <file.jsonl>`, rows with payload,
action and reply):

| Replies from | Replies | State a price | Changed by the guard |
|---|---|---|---|
| Gemma-4-31B-IT (4 dataset batches) | 167 | 49 | 0 |
| Three other models tried as dataset teachers | 299 | 107 | 1 |

The one change was a genuine error: the engine said 150 coins and the reply said "fifty coins".
The guard did not raise a false alarm on any Gemma reply, but Gemma also made no price mistake in
these 49 cases, so this shows the guard is safe, not that it catches many errors. The batches are
generated data that is not part of the repository (`data/` is git-ignored).

**Limits of this validation.**
- `facts` in the evaluation and the runtime guard only check the **price**. The evaluation covers
  the 4 of 15 scenarios that involve a purchase. Items and NPC state (for example "is the NPC
  angry") are not checked automatically.
- The guard reads amounts followed by a coin word. A price written without it ("it is twenty-five")
  is not seen.
- A corrected reply is a plain, flat line. It trades style for correctness on the rare occasion it
  fires.

```powershell
python test\eval_llm.py --amd --shots 2 --variants 5 --repeats 2 --verbose
```

## 4. Latency and error rate

**Method.** `test/bench_api.py` sends real engine prompts (the 15 scenarios of `eval_llm.py`,
repeated) to Gemma-4-31B-IT using the same settings as `create_engine()` for the API. "End to
end" includes waiting for a request slot and any retries; "network" is each HTTP attempt only.

| Run (30 calls each) | End-to-end median | End-to-end P95 | Max | Errors | Throughput |
|---|---|---|---|---|---|
| 1 worker | 5.7 s | 19.0 s | 24.5 s | 0 / 30 | 8.2 calls/min |
| 4 workers | 13.0 s | 67.7 s | 86.9 s | 0 / 30 | 10.6 calls/min |

Per HTTP attempt (network only): 1 worker, median 5.7 s, P95 17.9 s; 4 workers, median 7.6 s, P95
30.5 s, with 36 attempts for 30 calls.

**Caveats.** One run of each, on a shared free endpoint with no SLA, so numbers will vary with
load. They measure the whole path to a hosted model, **not the speed of the AMD hardware**. No
request failed permanently in 60 calls (0% error rate), but 6 attempts timed out in the 4-worker
run and were retried.

```powershell
python test\bench_api.py --calls 30 --workers 1 --label sequential --out docs\benchmarks\sequential.json
python test\bench_api.py --calls 30 --workers 4 --label concurrent-4 --out docs\benchmarks\concurrent4.json
```

## 5. Concurrency and API limits

**Limits of the free tier** (shared by the whole team's key): about 20 requests per minute, 8
concurrent requests, and 30 s timeouts in our client.

**What the engine does about them**
- It calls the model only when it matters: when a trigger fires, a memory event is stored, the
  intent is not "other", or the line has more than two words. Short filler gets a deterministic line.
- The client spaces request starts (3.2 s apart) so several threads share one rate limit, retries
  HTTP 429/5xx and timeouts with backoff and honours `Retry-After`.
- If the call still fails, the NPC answers with a canned line for its mood and the game continues.

**What was measured.** Going from 1 to 4 workers raised throughput only from 8.2 to 10.6 calls per
minute and made the slow tail much worse (P95 19 s to 68 s), because the endpoint queued the
requests and some exceeded the 30 s timeout. With this free endpoint, more parallelism did not help.

**What this means for many NPCs at once.** `process_action` is synchronous and the engine is not
thread-safe, so NPC requests are processed one after another. With replies taking a median of
about 6 s, a game that needs many simultaneous NPC lines would feel slow today. This is the main
limitation for real-time use.

**Planned.** An asynchronous request queue with a per-NPC priority, caching of repeated lines, and
the small local model from the fine-tuning plan as a fast offline path. None of these exist yet.

## 6. Reproducing everything

```powershell
python -m unittest discover -s test
python test\eval_labeled.py --reserve --llm
python test\bench_api.py --calls 30 --workers 1 --label sequential
python test\eval_llm.py --amd --shots 2 --variants 5 --repeats 2
```

The API commands need `AMD_API_KEY` in `.env` and use the team's shared quota (about 30 to 60
calls each).
