"""OpenRouter backend tests with a fake HTTP layer (no network, no real key)."""
import argparse
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "test"))

import modules.OpenRouterBackend as orb
from modules.OpenRouterBackend import DEFAULT_MODEL, OpenRouterBackend, OpenRouterError, load_key
from training.generate_dataset import make_teacher

PAYLOAD = ["World: Tavern", "NPC: Oswin", "You are Oswin, the Blacksmith.",
           "You are at the Forge.", "Style: neutral and businesslike."]
KEY = "sk-or-test-SECRET"


class FakeResponse:
    def __init__(self, content):
        self._body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(code, headers=None, body=b'{"error":"x"}'):
    return urllib.error.HTTPError("u", code, "err", headers or {}, io.BytesIO(body))


class FakeNet:
    """Plays a script of responses/exceptions and records the requests."""
    def __init__(self, *script):
        self.script, self.requests = list(script), []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)


def backend(net, sleeps=None, **kw):
    sleeps = sleeps if sleeps is not None else []
    return OpenRouterBackend(api_key=KEY, opener=net, sleep=sleeps.append, min_interval=0, **kw), sleeps


class OpenRouterTests(unittest.TestCase):
    def test_request_shape(self):
        net = FakeNet("Hmm. What do you need?")
        b, _ = backend(net)
        self.assertEqual(b.generate(PAYLOAD, "greet the NPC"), "Hmm. What do you need?")
        req = net.requests[0]
        self.assertEqual(req.full_url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(req.get_header("Authorization"), f"Bearer {KEY}")
        body = json.loads(req.data)
        self.assertEqual(body["model"], DEFAULT_MODEL)
        self.assertEqual(body["reasoning"], {"enabled": False})
        roles = [m["role"] for m in body["messages"]]
        self.assertEqual(roles[0], "system")
        self.assertEqual(roles[-1], "user")
        self.assertGreater(roles.count("assistant"), 0)            # few-shot teacher prompt
        self.assertNotIn("/no_think", body["messages"][-1]["content"])  # Qwen-only trick

    def test_missing_key_and_key_never_exposed(self):
        old, old_root = os.environ.pop("OPENROUTER_API_KEY", None), orb.ROOT
        try:
            with tempfile.TemporaryDirectory() as empty:
                orb.ROOT = empty  # no .env here, whatever the real project has
                with self.assertRaises(OpenRouterError):
                    OpenRouterBackend()
        finally:
            orb.ROOT = old_root
            if old is not None:
                os.environ["OPENROUTER_API_KEY"] = old
        b, _ = backend(FakeNet())
        self.assertNotIn(KEY, repr(b))

    def test_cleans_reasoning_and_prefix(self):
        b, _ = backend(FakeNet("<think>plan</think> Oswin: \"Get out.\""))
        self.assertEqual(b.generate(PAYLOAD, "attack the NPC"), "Get out.")

    def test_retries_rate_limit_with_backoff_and_retry_after(self):
        net = FakeNet(http_error(429, {"Retry-After": "7"}), http_error(503), "Fine, take it.")
        b, sleeps = backend(net)
        self.assertEqual(b.generate(PAYLOAD, "buy a lamp"), "Fine, take it.")
        self.assertEqual(len(net.requests), 3)
        self.assertEqual(sleeps, [7.0, 4.0])  # Retry-After first, then doubled backoff

    def test_gives_up_after_retries(self):
        net = FakeNet(*[http_error(429)] * 3)
        b, _ = backend(net, retries=2)
        with self.assertRaises(OpenRouterError):
            b.generate(PAYLOAD, "greet the NPC")
        self.assertEqual(len(net.requests), 3)

    def test_auth_errors_are_not_retried(self):
        net = FakeNet(http_error(401))
        b, sleeps = backend(net)
        with self.assertRaisesRegex(OpenRouterError, "HTTP 401"):
            b.generate(PAYLOAD, "greet the NPC")
        self.assertEqual((len(net.requests), sleeps), (1, []))

    def test_retries_without_reasoning_field_on_400(self):
        net = FakeNet(http_error(400), "Move along.")
        b, _ = backend(net)
        self.assertEqual(b.generate(PAYLOAD, "greet the NPC"), "Move along.")
        self.assertIn("reasoning", json.loads(net.requests[0].data))
        self.assertNotIn("reasoning", json.loads(net.requests[1].data))

    def test_empty_content_returns_empty_string(self):
        b, _ = backend(FakeNet(None))
        self.assertEqual(b.generate(PAYLOAD, "greet the NPC"), "")

    def test_spacing_between_calls(self):
        sleeps = []
        b = OpenRouterBackend(api_key=KEY, opener=FakeNet("a b c", "d e f"), sleep=sleeps.append, min_interval=5)
        b.generate(PAYLOAD, "greet the NPC")
        b.generate(PAYLOAD, "greet the NPC")
        self.assertTrue(sleeps and 0 < sleeps[-1] <= 5)


class DotEnvTests(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.pop("OPENROUTER_API_KEY", None)

    def tearDown(self):
        if self._old is not None:
            os.environ["OPENROUTER_API_KEY"] = self._old
        else:
            os.environ.pop("OPENROUTER_API_KEY", None)

    def write_env(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".env", delete=False, encoding="utf-8")
        f.write(text)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_reads_only_the_key_from_dotenv(self):
        path = self.write_env("# comment\nOTHER=1\nOPENROUTER_API_KEY=sk-or-abc123\n")
        self.assertEqual(load_key(path), "sk-or-abc123")
        self.assertNotIn("OTHER", os.environ)

    def test_quotes_export_and_bom(self):
        self.assertEqual(load_key(self.write_env('export OPENROUTER_API_KEY="sk-or-q"\n')), "sk-or-q")
        self.assertEqual(load_key(self.write_env("\ufeffOPENROUTER_API_KEY='sk-or-b'\r\n")), "sk-or-b")

    def test_environment_wins_over_dotenv(self):
        os.environ["OPENROUTER_API_KEY"] = "from-env"
        self.assertEqual(load_key(self.write_env("OPENROUTER_API_KEY=from-file\n")), "from-env")

    def test_missing_file_or_key(self):
        self.assertEqual(load_key("does-not-exist.env"), "")
        self.assertEqual(load_key(self.write_env("OTHER=1\n")), "")


class TeacherSelectionTests(unittest.TestCase):
    def args(self, **kw):
        base = dict(teacher="openrouter", model=None, teacher_license="", temperature=0.7, min_interval=3.0)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_license_is_required_for_openrouter(self):
        os.environ["OPENROUTER_API_KEY"] = KEY
        with self.assertRaises(SystemExit):
            make_teacher(self.args())

    def test_default_model_and_label(self):
        os.environ["OPENROUTER_API_KEY"] = KEY
        teacher, label = make_teacher(self.args(teacher_license="see model card"))
        self.assertEqual(label, DEFAULT_MODEL)
        self.assertNotIn(KEY, repr(teacher))


class AMDBackendTests(unittest.TestCase):
    def make(self, net):
        return orb.AMDBackend(api_key=KEY, opener=net, sleep=lambda s: None, min_interval=0)

    def test_endpoint_model_and_no_reasoning_field(self):
        net = FakeNet("Welcome, friend.")
        b = self.make(net)
        self.assertEqual(b.generate(PAYLOAD, "greet the NPC"), "Welcome, friend.")
        req = net.requests[0]
        body = json.loads(req.data)
        self.assertEqual(req.full_url, orb.AMD_ENDPOINT)
        self.assertEqual(body["model"], "Gemma-4-31B-IT")
        self.assertNotIn("reasoning", body)  # the gateway answers 400 to reasoning.enabled
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertNotIn(KEY, repr(b))

    def test_reads_amd_key_from_env_file_with_spaces(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, ".env")
            with open(path, "w") as f:
                f.writelines(["OPENROUTER_API_KEY =other", chr(10), "AMD_API_KEY = rc-abc123", chr(10)])
            old = os.environ.pop("AMD_API_KEY", None)
            try:
                self.assertEqual(load_key(path, key_name="AMD_API_KEY"), "rc-abc123")
            finally:
                if old is not None:
                    os.environ["AMD_API_KEY"] = old

    def test_429_honours_retry_after(self):
        sleeps = []
        net = FakeNet(http_error(429, {"Retry-After": "60"}), "Hm.")
        b = orb.AMDBackend(api_key=KEY, opener=net, sleep=sleeps.append, min_interval=0)
        self.assertEqual(b.generate(PAYLOAD, "greet the NPC"), "Hm.")
        self.assertEqual(sleeps, [60.0])

    def test_make_teacher_amd(self):
        os.environ["AMD_API_KEY"] = KEY
        args = argparse.Namespace(teacher="amd", model=None, teacher_license="see model card",
                                  temperature=0.7, min_interval=None)
        teacher, label = make_teacher(args)
        self.assertEqual(label, "Gemma-4-31B-IT")
        self.assertIsInstance(teacher, orb.AMDBackend)
        self.assertEqual(teacher.min_interval, 3.2)


if __name__ == "__main__":
    unittest.main()
