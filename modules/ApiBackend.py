"""
Hosted-model backends over an OpenAI-compatible chat-completions API.

ChatAPIBackend holds the shared code (HTTP, retries, rate limiting, key loading). Two concrete
backends configure it:

    AMDBackend         Gemma-4-31B-IT (default, any catalog model) on AMD GPUs through AMD Token Factory (key AMD_API_KEY)
    OpenRouterBackend  models on OpenRouter, used as a teacher for dataset generation
                       (key OPENROUTER_API_KEY)

Same interface as LlamaCppBackend, so engine and dataset generator can swap models. Uses only the
standard library. Keys are read from the environment or the project's .env file and are never
printed, stored or put in a dataset row.
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional

from modules.LLMLayer import LLMBackend, build_messages, clean_output

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENROUTER_KEY_NAME = "OPENROUTER_API_KEY"


class APIError(RuntimeError):
    pass


def load_key(env_file: Optional[str] = None, key_name: str = OPENROUTER_KEY_NAME) -> str:
    """API key from the environment, else from the project's .env file ('' if neither).

    Only `key_name` is read from .env; nothing else is exported and the value is never logged.
    """
    key = os.environ.get(key_name, "").strip()
    if key:
        return key
    path = env_file or os.path.join(ROOT, ".env")
    try:
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line.startswith("export "):
                    line = line[len("export "):]
                name, sep, value = line.partition("=")
                if sep and name.strip() == key_name:
                    return value.strip().strip("\"'")
    except OSError:
        pass
    return ""


class ChatAPIBackend(LLMBackend):
    """Shared client. Subclasses set `endpoint`, `key_name` and `title`."""
    endpoint = ""
    key_name = ""
    title = "Affekt Core"

    def __init__(self, model: str, api_key: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 160, min_interval: float = 3.0,
                 retries: int = 6, disable_reasoning: bool = False, timeout: float = 120.0,
                 opener: Optional[Callable] = None, sleep: Callable[[float], None] = time.sleep):
        self._key = api_key or load_key(key_name=self.key_name)
        if not self._key:
            raise APIError(f"{self.key_name} not found: set it in the environment or in .env")
        self.model = model
        self.name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.min_interval = min_interval
        self.retries = retries
        self.disable_reasoning = disable_reasoning
        self.shots = 3
        self._open = opener or urllib.request.urlopen
        self._sleep = sleep
        self.timeout = timeout
        self._last_call = 0.0  # start time of the latest slot handed out
        self._slot_lock = threading.Lock()  # lets several worker threads share one rate limit

    def __repr__(self) -> str:  # never expose the key
        return f"{type(self).__name__}(model={self.model!r})"

    # -- HTTP -------------------------------------------------------------
    def _request(self, body: Dict) -> urllib.request.Request:
        return urllib.request.Request(
            self.endpoint, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json",
                     "X-Title": self.title})

    def _post(self, body: Dict) -> Dict:
        delay = 2.0
        for attempt in range(self.retries + 1):
            self._wait_for_slot()
            try:
                with self._open(self._request(body), timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code in RETRY_STATUS and attempt < self.retries:
                    retry_after = e.headers.get("Retry-After") if e.headers else None
                    self._sleep(float(retry_after) if retry_after and retry_after.isdigit() else delay)
                    delay = min(delay * 2, 60)
                    continue
                raise APIError(f"HTTP {e.code}: {self._error_text(e)}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.retries:
                    self._sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                raise APIError(f"network error: {e}") from None
        raise APIError("retries exhausted")

    def _wait_for_slot(self) -> None:
        """Reserve the next request start time, at least `min_interval` after the previous one.

        With several threads the starts are spread out, so the per-minute limit holds while
        the slow responses overlap.
        """
        with self._slot_lock:
            now = time.time()
            start = max(now, self._last_call + self.min_interval)
            self._last_call = start
        if start > now:
            self._sleep(start - now)

    @staticmethod
    def _error_text(e: urllib.error.HTTPError) -> str:
        try:
            return e.read().decode("utf-8", "ignore")[:300]
        except Exception:
            return ""

    # -- generation -------------------------------------------------------
    def generate(self, payload: List[str], action: str) -> str:
        return clean_output(self.complete(build_messages(payload, action, no_think=False, shots=self.shots)))

    def complete(self, messages: List[Dict], max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        """Raw chat completion: the model's text for any messages (no NPC prompt, no cleaning)."""
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if self.disable_reasoning:
            body["reasoning"] = {"enabled": False}
        try:
            data = self._post(body)
        except APIError as e:
            # Some models reject the reasoning field; retry once without it.
            if self.disable_reasoning and str(e).startswith("HTTP 400"):
                body.pop("reasoning")
                data = self._post(body)
            else:
                raise
        try:
            text = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            raise APIError(f"unexpected response: {str(data)[:200]}") from None
        return text


class OpenRouterBackend(ChatAPIBackend):
    """OpenRouter models. Used as a dataset teacher; asks the model not to spend tokens on reasoning."""
    endpoint = OPENROUTER_ENDPOINT
    key_name = OPENROUTER_KEY_NAME
    title = "Affekt Core dataset generation"
    provider = "OpenRouter"

    def __init__(self, model: str = OPENROUTER_DEFAULT_MODEL, disable_reasoning: bool = True, **kw):
        super().__init__(model, disable_reasoning=disable_reasoning, **kw)


# -- AMD Token Factory (AMD Radeon Cloud, OpenAI-compatible, free shared endpoints) ----------
AMD_ENDPOINT = "https://developer.amd.com.cn/radeon/api/v1/chat/completions"
# Gemma-4-31B-IT is the default. It was off the Token Factory catalog on 2026-10-15 (demand after Google's announcement)
# and back on 2026-10-16, but loaded: some calls take 15-50 s or get HTTP 503 "no available workers". The game server
# therefore puts MiMo-V2.6-Flash (same service, 3-7 s) behind it (see docs/amd.md).
AMD_DEFAULT_MODEL = "Gemma-4-31B-IT"
AMD_KEY_NAME = "AMD_API_KEY"


class AMDBackend(ChatAPIBackend):
    """Gemma-4-31B-IT (or any catalog model) served on AMD GPUs by Token Factory.

    Same chat-completions protocol as OpenRouter. Differences: the gateway rejects the
    `reasoning` object with HTTP 400 (Gemma and MiMo do not need it, so nothing is sent), and
    the free tier allows ~20 requests/minute per account, hence the 3.2 s default spacing.
    429s honour Retry-After through the inherited retry loop. The key is shared by the whole team.
    """
    endpoint = AMD_ENDPOINT
    key_name = AMD_KEY_NAME
    title = "Affekt Core"
    provider = "AMD Token Factory"

    def __init__(self, model: str = AMD_DEFAULT_MODEL, min_interval: float = 3.2, **kw):
        super().__init__(model, min_interval=min_interval, **kw)


def describe(backend) -> str:
    """Who answered, for the UI and the logs: 'AMD Token Factory · MiMo-V2.6-Flash'."""
    name = getattr(backend, "provider", type(backend).__name__)
    model = getattr(backend, "model", "")
    return f"{name} · {model}" if model else name


class FailoverBackend(LLMBackend):
    """Asks `backends` in order and returns the first answer: AMD first, another provider as a backup.

    A backend that raised (HTTP 429, timeout, network error) is skipped for `cooldown` seconds, so the
    next players go straight to the backup instead of each waiting for the failing service. An empty reply
    only moves on to the next backend for that one call. If every backend is cooling down they are all
    tried anyway, in order; if all fail, the last error is raised and the engine answers with its
    deterministic line for the mood. `last_provider()` names who answered the latest call of this thread.
    """

    def __init__(self, backends: List[LLMBackend], cooldown: float = 45.0,
                 clock: Callable[[], float] = time.monotonic):
        if not backends:
            raise ValueError("FailoverBackend needs at least one backend")
        self.backends = list(backends)
        self.cooldown = cooldown
        self._clock = clock
        self._down: Dict[int, float] = {}  # id(backend) -> monotonic time until which it is skipped
        self._lock = threading.Lock()
        self._local = threading.local()
        self._shots = getattr(self.backends[0], "shots", 3)

    @property
    def shots(self) -> int:
        return self._shots

    @shots.setter
    def shots(self, value: int) -> None:  # the engine sets this once; every backend must use the same prompt
        self._shots = value
        for b in self.backends:
            b.shots = value

    def clear_last(self) -> None:
        self._local.provider = None

    def last_provider(self) -> Optional[str]:
        return getattr(self._local, "provider", None)

    def generate(self, payload: List[str], action: str) -> str:
        now = self._clock()
        with self._lock:
            ready = [b for b in self.backends if self._down.get(id(b), 0.0) <= now]
        order = ready + [b for b in self.backends if b not in ready]  # cooling-down ones go last, never dropped
        self._local.provider = None
        last_error: Exception = APIError("empty reply")
        for b in order:
            try:
                text = b.generate(payload, action)
            except Exception as e:  # noqa: BLE001 - any failure of one provider falls through to the next
                last_error = e
                with self._lock:
                    self._down[id(b)] = self._clock() + self.cooldown
                continue
            if text and text.strip():
                with self._lock:
                    self._down.pop(id(b), None)
                self._local.provider = describe(b)
                return text
            last_error = APIError("empty reply")
        raise last_error
