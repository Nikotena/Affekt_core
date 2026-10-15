# How Affekt Core uses AMD hardware

## Short version (for the submission form)

Affekt Core is a narrative engine that gives game NPCs emotion, memory and context. Plain code
decides what each NPC feels, remembers and knows, and a language model only phrases the final
line. That language model is **Gemma-4-31B-IT, served on AMD GPUs through AMD Token Factory
(AMD Radeon Cloud)**. The engine calls it through an OpenAI-compatible API to speak every NPC
line, to read player messages the rule-based parser cannot classify (it raised intent accuracy on
held-out lines from 54% to 64%), and to write the example dialogue for a future fine-tune of a
small local model. Because the model only phrases facts that the engine already chose, prompts
stay small and a failed call degrades to a canned line instead of breaking the game, which makes a
shared cloud endpoint a good fit for it.

## Details

**What runs on AMD**

| Use | Where in the code | Notes |
|---|---|---|
| NPC dialogue | `AMDBackend.generate`, used by `create_engine()` | Gemma-4-31B-IT, about 10 fact lines per prompt, 2 few-shot examples |
| Intent fallback | `modules/IntentLLM.py`, `create_engine(llm_intents=True)` | Only called for lines the rules label "other"; off by default |
| Dataset teacher | `training/generate_dataset.py --teacher amd` | Resumable, runs several requests in parallel; 120 sample rows generated so far |

**Why this design suits a hosted AMD endpoint**
- The engine never lets the model change game state, so a slow or unavailable model cannot corrupt
  the world; the client gives up after 30 s and the NPC answers with a line for its mood.
- Facts such as prices come from the engine and are put in the prompt ("Say the price: 25
  coins."), so the model's job is wording, not remembering.
- The client is a small standard-library wrapper around the chat-completions endpoint
  (`modules/OpenRouterBackend.py`), so no extra dependencies are needed to run on AMD Token Factory.

**What we measured on the free tier** (shared team key)
- Replies took from a few seconds to under a minute depending on load; in one rehearsal 2 of 14
  calls timed out and fell back to canned lines. A recorded run (`demo_run.json`, played with
  `python scene.py --replay demo_run.json`) is the safety net for live demos.
- Limits are about 20 requests per minute and 8 concurrent for the whole team's key.

**Not done yet**
- Training or fine-tuning on AMD GPUs (ROCm). The plan is in [finetune_compact.md](finetune_compact.md).
- Latency benchmarks of the model itself; the numbers above are end-to-end times through the shared
  free endpoint, not a measure of the hardware.
