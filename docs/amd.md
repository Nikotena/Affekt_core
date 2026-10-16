# How Affekt Core uses AMD hardware

Affekt Core is a narrative engine that gives game NPCs emotion, memory, and context. Plain code decides what each NPC feels, remembers, and knows, while a language model only phrases the final line. The default language model is **Gemma-4-31B-IT**, served on AMD GPUs through **AMD Token Factory**, with **MiMo-V2.6-Flash** (same service) as the first backup. The measurements in this repository (latency, price checks, intent fallback) were taken with Gemma-4-31B-IT. Gemma was missing from the Token Factory catalog on 2026-10-15 (demand after Google's announcement; the endpoint answered HTTP 400) and came back on 2026-10-16, but it is heavily loaded, which is why the demo has a backup chain.

The engine calls it through an OpenAI-compatible API to:
1. Speak every NPC line.
2. Read player messages that the rule-based parser cannot classify (raising intent accuracy on held-out lines from 54% to 64%).
3. Generate example dialogue to build a dataset for future fine-tuning of a small local model.

Because the model only phrases facts that the engine has already selected, prompts remain small. Furthermore, if a call fails or times out, the system safely degrades to a canned line instead of breaking the game. This design makes a shared cloud endpoint a reliable fit.

---

## Details

### What runs on AMD

| Use | Where in the code | Notes |
|---|---|---|
| **NPC dialogue** | `ApiBackend.generate`, used by `create_engine()` | Gemma-4-31B-IT by default, MiMo-V2.6-Flash as backup; ~10 fact lines per prompt, 2 few-shot examples |
| **Intent fallback** | `modules/IntentLLM.py`, `create_engine(llm_intents=True)` | Only called for lines labeled as "other" by rules; off by default |
| **Dataset teacher** | `training/generate_dataset.py --teacher amd` | Resumable, runs parallel requests; 120 sample rows generated so far |

### Why this design suits a hosted AMD endpoint

- **State Isolation:** The engine never allows the model to alter game state. A slow or unavailable model cannot corrupt the world; the client times out after 30 seconds, and the NPC responds with a canned line matching its current mood.
- **Fact Injection:** Specific facts (such as prices) originate from the engine and are injected into the prompt (*"Say the price: 25 coins."*). The model’s role is purely phrasing, not recall.
- **Lightweight Integration:** The client uses a minimal standard-library wrapper around the chat-completions endpoint (`modules/ApiBackend.py`), eliminating external dependencies to run on AMD Token Factory.

### Observations on the Free Tier (Shared Team Key)

- **Latency:** Response times ranged from a few seconds to under a minute depending on server load. During one rehearsal, 2 out of 14 calls timed out and fell back to canned responses. A recorded playback (`demo_run.json`, executed via `python scene.py --replay demo_run.json`) serves as a safety net for live demonstrations.
- **Rate Limits:** Shared team key limits are approximately 20 requests per minute with a maximum of 8 concurrent requests.

### What runs on AMD in the live demo

Every NPC line of the public demo is written by a model on **AMD Token Factory**. The game server
(`Kankei_game/server/server.py`) asks, in order, and the first one that answers wins:

1. **Gemma-4-31B-IT on AMD Token Factory** (`AFFEKT_AMD_MODEL`).
2. **MiMo-V2.6-Flash on AMD Token Factory** (`AFFEKT_AMD_BACKUP_MODEL`): also AMD, and not as saturated.
3. **Gemma 4 on OpenRouter**, only if an `OPENROUTER_API_KEY` exists: last resort, outside AMD.

The deterministic part (emotion, memory, graph, intent, price guard) runs on the web server itself. The panel next to
the game shows which provider wrote each line, e.g. `LLM: AMD Token Factory · Gemma-4-31B-IT`.

**Why a chain.** On 2026-10-16 Gemma-4-31B-IT answered in 2-10 s on some calls and took 50 s or returned
`503 no available workers` on others, while MiMo answered in 3-7 s every time. The first two steps wait at most 15 s
with no retry; a step that fails is skipped for 45 s so the next players do not wait on it. Only the last step keeps the
full patience (30 s, two retries); if everything fails the NPC answers with its canned line for the mood.
`AFFEKT_FALLBACK=0` keeps only the first step (to record a take with one model) and `AFFEKT_AMD_MODEL=MiMo-V2.6-Flash`
makes MiMo the first choice, for example if Gemma is overloaded during judging. The shared AMD key allows about 20
requests per minute and 8 concurrent.

### MiMo-V2.6-Flash as the backup model (2026-10-15)

While Gemma was off the catalog, six of its 11 models were tried through the real engine (same NPC prompt,
2 few-shot examples) on a price question and an insult to Marta. Single runs, not a benchmark:

| Model | Result |
|---|---|
| **MiMo-V2.6-Flash** (chosen as backup) | Real model lines in all 6 calls, correct price, about 4-7 s; on the 4-line arc (price, insult, haggle, pay) the lines stayed in character and the price guard returned `ok` each time |
| DeepSeek-V4.1-Flash | Good lines, 2-14 s |
| Qwen3.8-27B, Qwen3.8-Flash-Next | One of two replies each fell back to the canned line |
| GLM-5.3-Flash | Good lines but 27-76 s |
| MiniCPM5-2B | Leaked its reasoning ("Okay, the user is Marta...") into the reply |

The other catalog models are vision or document models and were not tried. Pick another with
`AMDBackend(model=...)`; `AFFEKT_AMD_MODEL` does the same in the game's bridge.

### Future Work / Next Steps

- **Fine-Tuning:** Training or fine-tuning models directly on AMD GPUs via ROCm (detailed plan in [finetune_compact.md](finetune_compact.md)).
- **Hardware Benchmarks:** Conducting isolated hardware latency benchmarks for the model itself (current figures reflect end-to-end latency across the shared free endpoint).

Measured numbers and how to reproduce them: [evaluation.md](evaluation.md).
