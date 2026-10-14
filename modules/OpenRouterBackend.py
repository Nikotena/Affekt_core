"""
OpenRouter backend (teacher for dataset generation).

Same interface as LlamaCppBackend, so the dataset generator can swap teachers. Uses only
the standard library. The API key is read from the OPENROUTER_API_KEY environment
variable or the project's .env file and is never printed, stored or put in a row.
"""

import json
import os
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


def load_key(env_file: Optional[str] = None) -> str:
    """API key from the environment, else from the project's .env file ('' if neither).

    Only KEY_NAME is read from .env; nothing else is exported and the value is never logged.
    """
    key = os.environ.get(KEY_NAME, "").strip()
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
                if sep and name.strip() == KEY_NAME:
                    return value.strip().strip("\"'")
    except OSError:
        pass
    return ""


class OpenRouterBackend(LLMBackend):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 160, min_interval: float = 3.0,
                 retries: int = 6, disable_reasoning: bool = True,
                 opener: Optional[Callable] = None, sleep: Callable[[float], None] = time.sleep):
        self._key = api_key or load_key()
        if not self._key:
            raise OpenRouterError(f"{KEY_NAME} not found: set it in the environment or in .env")
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
        self._last_call = 0.0

    def __repr__(self) -> str:  # never expose the key
        return f"OpenRouterBackend(model={self.model!r})"

    # -- HTTP -------------------------------------------------------------
    def _request(self, body: Dict) -> urllib.request.Request:
        return urllib.request.Request(
            ENDPOINT, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json",
                     "X-Title": "Kankei Core dataset generation"})

    def _post(self, body: Dict) -> Dict:
        wait = self.min_interval - (time.time() - self._last_call)
        if wait > 0:
            self._sleep(wait)
        delay = 2.0
        for attempt in range(self.retries + 1):
            try:
                with self._open(self._request(body), timeout=120) as resp:
                    self._last_call = time.time()
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                self._last_call = time.time()
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

    @staticmethod
    def _error_text(e: urllib.error.HTTPError) -> str:
        try:
            return e.read().decode("utf-8", "ignore")[:300]
        except Exception:
            return ""

    # -- generation -------------------------------------------------------
    def generate(self, payload: List[str], action: str) -> str:
        body = {
            "model": self.model,
            "messages": build_messages(payload, action, no_think=False, shots=self.shots),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
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
        return clean_output(text)
