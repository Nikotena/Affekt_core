"""create_engine(): the one-call demo setup, and the faster mood recovery."""
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import AffektEngine as ke
from modules.IntentParser import parse
from modules.LLMLayer import StubBackend, set_example_variants
from modules.VectorialPersonality import VectorialPersonality


class FakeLlama:
    """Stands in for LlamaCppBackend so no model is loaded."""
    instances = []

    def __init__(self, path, **kwargs):
        self.path, self.kwargs, self.shots = path, kwargs, 3
        FakeLlama.instances.append(self)

    def generate(self, payload, action):
        return "Fake reply."


class FactoryTests(unittest.TestCase):
    def setUp(self):
        FakeLlama.instances.clear()
        self.addCleanup(set_example_variants, 0)  # the factory sets module-level state
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fake_model = os.path.join(self.tmp.name, "gemma-4-test.gguf")
        open(self.fake_model, "wb").write(b"x")

    def test_injected_backend_gives_a_running_engine(self):
        e = ke.create_engine(llm=StubBackend())
        self.assertTrue(e.running)
        self.assertIn("response", e.process_action("NPC_1", "Hey Marta, how much for a potion?"))

    def test_measured_configuration_is_applied(self):
        with mock.patch.object(ke, "LlamaCppBackend", FakeLlama), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ke.GPU_LAYERS_ENV, None)
            e = ke.create_engine(model_path=self.fake_model)
        llm = FakeLlama.instances[0]
        self.assertEqual(llm.path, self.fake_model)
        self.assertEqual((llm.kwargs["temperature"], llm.kwargs["top_p"], llm.kwargs["top_k"]), (0.5, 0.9, 40))
        self.assertEqual((llm.kwargs["prompt_cache_mb"], llm.kwargs["n_gpu_layers"], llm.shots), (512, 0, 2))
        self.assertIs(e.llm, llm)
        # 5 fixed example sets per mood -> several distinct prompt prefixes, not one and not random
        from modules.LLMLayer import select_few_shot
        sets = {tuple(r for _f, _a, r in select_few_shot("hostile", limit=2)) for _ in range(80)}
        self.assertEqual(len(sets), 5)

    def test_gpu_layers_from_argument_or_environment(self):
        with mock.patch.object(ke, "LlamaCppBackend", FakeLlama):
            ke.create_engine(model_path=self.fake_model, n_gpu_layers=-1)
            with mock.patch.dict(os.environ, {ke.GPU_LAYERS_ENV: "20"}):
                ke.create_engine(model_path=self.fake_model)
        self.assertEqual([i.kwargs["n_gpu_layers"] for i in FakeLlama.instances], [-1, 20])

    def test_missing_model_fails_loudly(self):
        with self.assertRaises(FileNotFoundError) as cm:
            ke.create_engine(model_path=os.path.join(self.tmp.name, "nope.gguf"))
        self.assertIn("nope.gguf", str(cm.exception))
        with mock.patch.object(ke, "find_default_model", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "AFFEKT_MODEL"):
                ke.create_engine(use_amd_api=False)

    def test_api_is_chosen_automatically_only_with_a_key(self):
        from modules.OpenRouterBackend import AMDBackend
        with mock.patch.dict(os.environ, {"AMD_API_KEY": "rc-test"}):
            e = ke.create_engine()
            self.assertIsInstance(e.llm, AMDBackend)
            self.assertEqual((e.llm.timeout, e.llm.retries), (30.0, 2))
            with self.assertRaises(FileNotFoundError):  # an explicit local model path wins
                ke.create_engine(model_path=os.path.join(self.tmp.name, "nope.gguf"))

    def test_api_failure_falls_back_to_canned_line(self):
        class Down(StubBackend):
            def generate(self, payload, action):
                raise RuntimeError("HTTP 503")
        e = ke.create_engine(llm=Down())
        r = e.process_action("NPC_1", "attack the NPC!")
        self.assertTrue(r["response"])

    def test_default_model_search(self):
        with mock.patch.dict(os.environ, {ke.MODEL_ENV: "/somewhere/model.gguf"}):
            self.assertEqual(ke.find_default_model(), "/somewhere/model.gguf")
        env = {k: v for k, v in os.environ.items() if k != ke.MODEL_ENV}
        with mock.patch.dict(os.environ, env, clear=True):
            os.makedirs(os.path.join(self.tmp.name, "model", "sub"))
            small = os.path.join(self.tmp.name, "model", "gemma-4-small.gguf")
            q4 = os.path.join(self.tmp.name, "model", "sub", "gemma-4-E2B-it-Q4_0.gguf")
            open(small, "wb").write(b"x")
            open(q4, "wb").write(b"x")
            with mock.patch.object(ke, "ROOT", self.tmp.name):
                self.assertIsNone(ke.find_default_model())          # both are tiny: truncated downloads
                with mock.patch.object(ke.os.path, "getsize", return_value=3_000_000_000):
                    self.assertEqual(ke.find_default_model(), q4)    # Q4_0 preferred over other Gemma files


class MoodRecoveryTests(unittest.TestCase):
    def calm_after_insults(self, n):
        p = VectorialPersonality("n")
        for _ in range(n):
            p.update_from_action(list(parse("You're a pathetic fool.").delta))
        self.assertEqual(p.mood(), "hostile")
        turns = 0
        while p.mood() in ("hostile", "tense") and turns < 60:
            p.update_from_action(list(parse("Interesting weather we're having.").delta))
            turns += 1
        return turns

    def test_an_angry_npc_calms_down_in_a_reasonable_time(self):
        self.assertLessEqual(self.calm_after_insults(2), 5)
        self.assertLessEqual(self.calm_after_insults(5), 10)


if __name__ == "__main__":
    unittest.main()
