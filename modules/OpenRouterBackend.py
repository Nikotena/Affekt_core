"""
OpenRouter backend (teacher for dataset generation).

Same interface as LlamaCppBackend, so the dataset generator can swap teachers. Uses only
the standard library. The API key is read from the OPENROUTER_API_KEY environment
variable or the project's .env file and is never printed, stored or put in a row.
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional

from modules.LLMLayer import LLMBackend, build_messages, clean_output

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY_NAME = "OPENROUTER_API_KEY"


class OpenRouterError(RuntimeError):
    pass


def load_key(env_file: Optional[str] = None, key_name: str = KEY_NAME) -> str:
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


class OpenRouterBackend(LLMBackend):
    endpoint = ENDPOINT
    key_name = KEY_NAME
    title = "Affekt Core dataset generation"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 160, min_interval: float = 3.0,
                 retries: int = 6, disable_reasoning: bool = True, timeout: float = 120.0,
                 opener: Optional[Callable] = None, sleep: Callable[[float], None] = time.sleep):
        self._key = api_key or load_key(key_name=self.key_name)
        if not self._key:
            raise OpenRouterError(f"{self.key_name} not found: set it in the environment or in .env")
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
                raise OpenRouterError(f"HTTP {e.code}: {self._error_text(e)}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.retries:
                    self._sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                raise OpenRouterError(f"network error: {e}") from None
        raise OpenRouterError("retries exhausted")

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
        except OpenRouterError as e:
            # Some models reject the reasoning field; retry once without it.
            if self.disable_reasoning and str(e).startswith("HTTP 400"):
                body.pop("reasoning")
                data = self._post(body)
            else:
                raise
        try:
            text = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            raise OpenRouterError(f"unexpected response: {str(data)[:200]}") from None
        return text


# -- AMD Token Factory (AMD Radeon Cloud, OpenAI-compatible, free shared endpoints) ----------
AMD_ENDPOINT = "https://developer.amd.com.cn/radeon/api/v1/chat/completions"
AMD_DEFAULT_MODEL = "Gemma-4-31B-IT"
AMD_KEY_NAME = "AMD_API_KEY"


class AMDBackend(OpenRouterBackend):
    """Gemma-4-31B-IT (or any catalog model) served on AMD GPUs by Token Factory.

    Same chat-completions protocol. Differences from OpenRouter: the gateway rejects the
    `reasoning` object with HTTP 400 (Gemma does not think by default, so nothing is sent), and
    the free tier allows ~20 requests/minute per account, hence the 3.2 s default spacing.
    429s honour Retry-After through the inherited retry loop. The key is shared by the whole team.
    """
    endpoint = AMD_ENDPOINT
    key_name = AMD_KEY_NAME
    title = "Affekt Core"

    def __init__(self, model: str = AMD_DEFAULT_MODEL, min_interval: float = 3.2,
                 disable_reasoning: bool = False, **kw):
        super().__init__(model, min_interval=min_interval, disable_reasoning=disable_reasoning, **kw)
