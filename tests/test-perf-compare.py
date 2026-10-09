#!/usr/bin/env python3
"""Unit tests for paired performance sample comparisons."""

import json
import math
import unittest
from typing import cast

from perf_compare import compare_samples


class CompareSamplesTests(unittest.TestCase):
    def test_stable_identical_samples_pass(self):
        report = compare_samples([2.0] * 5, [2.0] * 5)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["paired_regressions"], 0)
        self.assertEqual(report["ratio_of_medians"], 1.0)
        self.assertEqual(report["median_added_seconds"], 0.0)

    def test_stable_improvement_passes(self):
        report = compare_samples([4.0] * 5, [1.0] * 5)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["paired_regressions"], 0)
        self.assertEqual(report["ratio_of_medians"], 0.25)

    def test_stable_large_slowdown_is_a_regression(self):
        report = compare_samples([1.0] * 5, [4.0] * 5)

        self.assertEqual(report["status"], "regression")
        self.assertEqual(report["paired_regressions"], 5)
        self.assertEqual(
            report["reason"], "median slowdown repeats across paired samples"
        )

    def test_four_of_five_slow_pairs_are_repeatable(self):
        report = compare_samples([1.0] * 5, [4.0, 4.0, 4.0, 4.0, 1.0])

        self.assertEqual(report["status"], "regression")
        self.assertEqual(report["paired_regressions"], 4)

    def test_single_extreme_candidate_outlier_does_not_trigger_regression(self):
        report = compare_samples([1.0] * 5, [1.0, 1.0, 1.0, 1.0, 100.0])

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["summaries"]["candidate"]["middle_spread_seconds"], 0.0)
        self.assertEqual(report["paired_regressions"], 1)

    def test_tiny_absolute_delay_passes_even_with_large_ratio(self):
        report = compare_samples([0.01] * 5, [0.1] * 5)

        self.assertEqual(report["status"], "passed")
        self.assertGreater(report["ratio_of_medians"], 3.0)
        self.assertLess(report["median_added_seconds"], 0.25)

    def test_noisy_reference_prevents_regression_decision(self):
        report = compare_samples([1.0, 1.0, 1.0, 4.0, 5.0], [4.0] * 5)

        self.assertEqual(report["status"], "inconclusive")
        self.assertIn("noise", report["reason"])
        self.assertIsNone(report["paired_regressions"])

    def test_noisy_candidate_prevents_regression_decision(self):
        report = compare_samples([1.0] * 5, [4.0, 4.0, 4.0, 10.0, 100.0])

        self.assertEqual(report["status"], "inconclusive")
        self.assertIn("noise", report["reason"])
        self.assertIsNone(report["paired_regressions"])

    def test_missing_unequal_and_insufficient_pairs_are_inconclusive(self):
        cases = (
            ([], [1.0] * 5, "contain samples"),
            ([1.0] * 5, [1.0] * 4, "counts differ"),
            ([1.0] * 4, [1.0] * 4, "five paired samples"),
        )
        for baseline, candidate, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                report = compare_samples(baseline, candidate)

                self.assertEqual(report["status"], "inconclusive")
                self.assertIn(expected_reason, report["reason"])

    def test_invalid_times_are_inconclusive_and_json_safe(self):
        for invalid in (math.nan, math.inf, -math.inf, 0.0, -1.0):
            with self.subTest(invalid=invalid):
                baseline = [1.0, 1.0, invalid, 1.0, 1.0]
                report = compare_samples(baseline, [1.0] * 5)

                self.assertEqual(report["status"], "inconclusive")
                if math.isfinite(invalid):
                    self.assertEqual(report["baseline_samples"][2], invalid)
                else:
                    self.assertIsNone(report["baseline_samples"][2])
                json.dumps(report, allow_nan=False)

    def test_strict_threshold_boundaries_pass(self):
        exact_ratio = compare_samples([1.0] * 5, [3.0] * 5)
        exact_added = compare_samples(
            [0.0625] * 5,
            [0.3125] * 5,
            min_added_seconds=0.25,
        )

        self.assertEqual(exact_ratio["status"], "passed")
        self.assertEqual(exact_ratio["ratio_of_medians"], 3.0)
        self.assertEqual(exact_ratio["paired_regressions"], 0)
        self.assertEqual(exact_added["status"], "passed")
        self.assertEqual(exact_added["median_added_seconds"], 0.25)
        self.assertEqual(exact_added["paired_regressions"], 0)

    def test_spread_at_noise_limit_is_not_noisy(self):
        report = compare_samples(
            [1.0, 1.0, 1.25, 1.5, 100.0],
            [1.25] * 5,
            max_relative_spread=0.4,
        )

        self.assertEqual(report["summaries"]["baseline"]["middle_spread_seconds"], 0.5)
        self.assertEqual(report["status"], "passed")

    def test_invalid_policy_raises_value_error(self):
        invalid_policies = (
            {"ratio": 1.0},
            {"ratio": math.nan},
            {"ratio": math.inf},
            {"min_added_seconds": -0.01},
            {"min_added_seconds": math.nan},
            {"min_added_seconds": math.inf},
            {"max_relative_spread": 0.0},
            {"max_relative_spread": math.nan},
            {"max_relative_spread": math.inf},
        )
        for policy in invalid_policies:
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                compare_samples([1.0] * 5, [1.0] * 5, **policy)

    def test_six_samples_need_five_slow_pairs(self):
        four = compare_samples(
            [1.0] * 6, [4.0] * 4 + [1.0] * 2, max_relative_spread=10.0
        )
        five = compare_samples([1.0] * 6, [4.0] * 5 + [1.0])
        self.assertEqual(four["paired_regressions"], 4)
        self.assertEqual(four["status"], "inconclusive")
        self.assertEqual(five["paired_regressions"], 5)
        self.assertEqual(five["status"], "regression")

    def test_two_slow_pairs_without_median_regression_are_inconclusive(self):
        report = compare_samples(
            [1.0, 1.0, 1.0, 3.0, 3.0],
            [4.0, 4.0, 1.0, 3.0, 3.0],
            max_relative_spread=10.0,
        )
        self.assertEqual(report["ratio_of_medians"], 3.0)
        self.assertEqual(report["paired_regressions"], 2)
        self.assertEqual(report["status"], "inconclusive")

    def test_invalid_candidate_sample_is_not_dropped(self):
        for value in (None, math.nan, math.inf, 0.0, -1.0, True, "1"):
            with self.subTest(value=value):
                candidate = cast(list[float], [1.0, value, 1.0, 1.0, 1.0])
                report = compare_samples([1.0] * 5, candidate)
                self.assertEqual(report["status"], "inconclusive")
                self.assertEqual(len(report["candidate_samples"]), 5)
                json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
