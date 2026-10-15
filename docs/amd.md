# How Affekt Core uses AMD hardware

Affekt Core is a narrative engine that gives game NPCs emotion, memory, and context. Plain code decides what each NPC feels, remembers, and knows, while a language model only phrases the final line. That language model is **Gemma-4-31B-IT**, served on AMD GPUs through **AMD Token Factory**. 

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
| **NPC dialogue** | `ApiBackend.generate`, used by `create_engine()` | Gemma-4-31B-IT, ~10 fact lines per prompt, 2 few-shot examples |
| **Intent fallback** | `modules/IntentLLM.py`, `create_engine(llm_intents=True)` | Only called for lines labeled as "other" by rules; off by default |
| **Dataset teacher** | `training/generate_dataset.py --teacher amd` | Resumable, runs parallel requests; 120 sample rows generated so far |

### Why this design suits a hosted AMD endpoint

- **State Isolation:** The engine never allows the model to alter game state. A slow or unavailable model cannot corrupt the world; the client times out after 30 seconds, and the NPC responds with a canned line matching its current mood.
- **Fact Injection:** Specific facts (such as prices) originate from the engine and are injected into the prompt (*"Say the price: 25 coins."*). The model’s role is purely phrasing, not recall.
- **Lightweight Integration:** The client uses a minimal standard-library wrapper around the chat-completions endpoint (`modules/ApiBackend.py`), eliminating external dependencies to run on AMD Token Factory.

### Observations on the Free Tier (Shared Team Key)

- **Latency:** Response times ranged from a few seconds to under a minute depending on server load. During one rehearsal, 2 out of 14 calls timed out and fell back to canned responses. A recorded playback (`demo_run.json`, executed via `python scene.py --replay demo_run.json`) serves as a safety net for live demonstrations.
- **Rate Limits:** Shared team key limits are approximately 20 requests per minute with a maximum of 8 concurrent requests.

### Future Work / Next Steps

- **Fine-Tuning:** Training or fine-tuning models directly on AMD GPUs via ROCm (detailed plan in [finetune_compact.md](finetune_compact.md)).
- **Hardware Benchmarks:** Conducting isolated hardware latency benchmarks for the model itself (current figures reflect end-to-end latency across the shared free endpoint).