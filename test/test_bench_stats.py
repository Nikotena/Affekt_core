"""The statistics helpers used by the benchmarks (so the published numbers can be trusted)."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "test"))

import bench_api  # noqa: E402
import eval_labeled  # noqa: E402


class PercentileTests(unittest.TestCase):
    def test_nearest_rank(self):
        values = list(range(1, 11))
        self.assertEqual(bench_api.percentile(values, 50), 5)
        self.assertEqual(bench_api.percentile(values, 95), 10)
        self.assertEqual(bench_api.percentile([7], 95), 7)

    def test_unsorted_input_and_empty(self):
        self.assertEqual(bench_api.percentile([9, 1, 5], 100), 9)
        self.assertNotEqual(bench_api.percentile([], 50), bench_api.percentile([], 50))  # NaN


class WilsonTests(unittest.TestCase):
    def test_known_interval(self):
        lo, hi = eval_labeled.wilson(32, 50)
        self.assertAlmostEqual(lo, 0.501, places=2)
        self.assertAlmostEqual(hi, 0.759, places=2)

    def test_edges_stay_inside_0_1(self):
        for right in (0, 50):
            lo, hi = eval_labeled.wilson(right, 50)
            self.assertTrue(0.0 <= lo <= hi <= 1.0)


if __name__ == "__main__":
    unittest.main()
