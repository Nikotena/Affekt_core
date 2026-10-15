# Fine-tuning a small local model for Affekt Core: compact plan

**Status: planned, not started (post-hackathon).** No model has been trained. The dataset pipeline
exists and has been tried at small scale. The thresholds under "Success metric" and "Guardrails"
are proposals to be confirmed before training.

## Why

Today the NPC line is written by **Gemma-4-31B-IT through the AMD Token Factory API**, with two
few-shot examples in the prompt. That gives good quality but needs a network connection, shares a
rate-limited team key (about 20 requests per minute) and replies in seconds to under a minute.

**Goal:** a small model (Qwen3-0.6B, with 1.7B as a comparison) that writes the NPC line in the
tone the engine asks for, **with no examples in the prompt**, running on a local CPU. It would be
the offline and on-device option; the API model stays the high-quality option.

**Who cares:** players (NPCs that sound alive and consistent, even offline); developers (low
latency and no per-call dependency on a hosted service).

**Decision owner:** Aaron.

**Action changed by the model:** `create_engine()` gains a third backend. The deterministic engine
(vector, memory, graph, triggers) does not change: it still decides the MOOD and the FACTS, and
the model only words them. Switching back is one argument: `create_engine(use_amd_api=True)` or
`model_path=`.

## Success metric (on held-out scenarios)

- Tone >= 85% (heuristic in `test/tone.py`) and >= 80% acceptable in a manual review of 30 samples.
- Median latency <= 2.0 s on a local CPU, with no few-shot examples.

## Guardrails (must not get worse than the base model)

- Facts (right price and item) >= 98%.
- Short replies (<= 3 sentences, <= 220 characters) >= 98%.
- No meta-language ("the player", "mood", "style") >= 98%.
- Variety across repeated runs >= 85%.
- No copying of training replies (4-gram overlap) >= 95%.
- Zero stage directions (`*...*`, brackets, parentheses).

**Unacceptable mistakes:** wrong price or item; breaking character (talking about the player in
third person or mentioning the system); content outside the game fiction.
**Acceptable mistakes:** flat but correct sentences; slightly soft tone.

## Assumptions

- The deterministic engine picks the mood correctly; the model does not correct it.
- The tone heuristic is only a guide (it has produced false passes and fails), hence the manual review.
- 2,000-3,000 clean examples are enough for LoRA on models this size (to be validated).

## Constraints

- Inference on a local CPU (llama.cpp, GGUF).
- Training hardware is not settled. AMD Developer Cloud GPU hours were requested but not granted,
  so this plan does not depend on them; a free cloud GPU (for example a T4) or any ROCm/CUDA machine
  would do, since the models are small.
- The teacher model's license must allow training on its outputs, and is recorded in every row.

## Data

- **Input:** the exact payload produced by `AffektEngine.build_payload`, the same structure used in
  the game.
- **Output:** the NPC line written by the teacher and filtered.
- **Teacher:** Gemma-4-31B-IT through AMD Token Factory (`--teacher amd`). Weights are served
  unmodified and the license is Apache-2.0, recorded per row with `--teacher-license`. Earlier
  small trials used other teachers (a free Nemotron model on OpenRouter, a local TBrilliance 2.6B
  at about 14 s per reply on CPU); the pipeline supports all of them (`--teacher amd|openrouter|local`).
- **Scenarios:** a grid of NPC (role and name) x mood x action x item and price. Items and prices
  vary widely; game items are included too.
- **Filter** (`training/filters.py`): reuses the benchmark checks and rejects stage directions,
  third person, name prefixes, reasoning leaks, quoted dialogue, profanity, replies that recite
  the NPC's own facts, invented prices or places, spelled-out prices that cannot be normalized,
  too-short replies and near-duplicates. Spelled-out numbers are first rewritten to digits.
- **Dataset version:** hash of the file plus the generation seed, stored next to the model.

**Where it stands:** 120 rows have been generated with the Gemma teacher and are not yet
filtered; earlier trial runs left a small filtered set (a few dozen training rows). That is far
below the 2,000-3,000 target.

**Throughput with the API:** with `--workers 6` the observed time per row ranged from about 4 s to
about 20 s depending on load, and the team's limit is about 20 requests per minute, so 2,000 rows
is roughly 2-4 hours of generation. The run is resumable, so an interrupted run continues where
it stopped.

## Split policy (against leakage)

- The scenarios in `test/eval_llm.py` never enter training.
- NPC roles and items that training never sees are reserved for evaluation (`eval_unseen`).
- Near-duplicate replies are removed within each split. A check across training and evaluation
  is not implemented yet and should be added before training.

## Train/serve parity

The training prompt is the inference prompt with **zero examples**: `build_messages(..., shots=0)`
(including `/no_think` for Qwen3). Samples are built from the raw rows when `filter` runs, so
after any change to the prompt (for example the recent rule that price questions must state the
price) re-run `filter` and retrain. The few-shot examples in `modules/LLMLayer.py` exist only for
the API and base-model paths.

## Baseline

Measured earlier on the base models with 3 few-shot examples, 5 repeats, seed 1. These numbers
predate several prompt and engine changes and should be re-measured before comparing. The
Gemma-4-31B-IT API baseline has not been measured yet (`python test/eval_llm.py --amd`).

| | Qwen3-0.6B | Qwen3-1.7B |
|---|---|---|
| Overall | 94% | 96% |
| Tone | 62% | 84% |
| Facts | 100% | 100% |
| Variety | 99% | 88% |
| Latency | 3.4 s | 5.4 s |

**Eval slices:** by mood (hostile / friendly / neutral / tense), by action (buy / attack / greet /
help / ask), and by NPC.

## Known risks

- The model learns the teacher's flaws (stage directions, quirks), hence the filter and the manual review.
- Overfitting to dataset phrases (copying), hence the 4-gram metric.
- A keyword tone check can reward vocabulary rather than attitude.
- Fine-tuning can erase general ability. That does not matter for this use, but check that the
  model does not produce junk on out-of-distribution input.

## How to run the pipeline

```powershell
# 1. Generate rows (resumable). The API key is read from .env.
python training\generate_dataset.py generate --teacher amd --teacher-license "Apache-2.0 (Gemma 4 weights, unmodified)" --n 1000 --workers 6 --out data\raw_amd.jsonl

# 2. Filter into train / eval_seen / eval_unseen chat samples and a rejection report
python training\generate_dataset.py filter --raw data\raw_amd.jsonl --out-dir data

# 3. Read a sample by hand before trusting the filter
python training\generate_dataset.py review --out-dir data
```

## Next experiment

Generate about 1,000 rows with the Gemma teacher, filter them, and read 30 by hand to confirm the
filter keeps good lines and drops bad ones. Note the share of rows kept per rejection reason. Only
then scale to the full set and train.

## Rollback or fallback

Keep the base model and the API path. `create_engine()` already falls back to a canned line for the
NPC's mood when a model fails, and the backend is a single argument, so going back means changing
it. The new model is adopted only if it meets **all** thresholds on the held-out scenarios.
