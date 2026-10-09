"""Compare paired timing samples with bounded noise and regression rules."""

import math
from typing import Any


def _validated_policy_value(
    name: str, value: float, *, minimum: float, strict: bool
) -> float:
    if isinstance(value, (bool, str, bytes)):
        number = math.nan
    else:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if number <= minimum if strict else number < minimum:
        relation = "greater than" if strict else "at least"
        raise ValueError(f"{name} must be {relation} {minimum}")
    return number


def _copy_samples(samples: object) -> list[object] | None:
    if not isinstance(samples, list):
        return None
    return list(samples)


def _json_safe_sample(sample: Any) -> object:
    if sample is None or isinstance(sample, (str, bool, int)):
        return sample
    try:
        value = float(sample)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _elapsed_seconds(sample: Any) -> float | None:
    if isinstance(sample, (bool, str, bytes)):
        return None
    try:
        value = float(sample)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def _median(sorted_values: list[float]) -> float:
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[middle]
    lower = sorted_values[middle - 1]
    upper = sorted_values[middle]
    return lower + (upper - lower) / 2


def _summarize(values: list[float]) -> dict[str, int | float]:
    sorted_values = sorted(values)
    trim = max(1, len(sorted_values) // 5)
    middle = sorted_values[trim : len(sorted_values) - trim]
    return {
        "count": len(sorted_values),
        "median_seconds": _median(sorted_values),
        "middle_spread_seconds": middle[-1] - middle[0],
    }


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def compare_samples(
    baseline: list[float],
    candidate: list[float],
    *,
    ratio: float = 3.0,
    min_added_seconds: float = 0.25,
    max_relative_spread: float = 0.5,
) -> dict[str, Any]:
    """Classify paired timings as passed, regression, or inconclusive."""
    ratio = _validated_policy_value("ratio", ratio, minimum=1, strict=True)
    min_added_seconds = _validated_policy_value(
        "min_added_seconds", min_added_seconds, minimum=0, strict=False
    )
    max_relative_spread = _validated_policy_value(
        "max_relative_spread", max_relative_spread, minimum=0, strict=True
    )

    baseline_original = _copy_samples(baseline)
    candidate_original = _copy_samples(candidate)
    report: dict[str, Any] = {
        "status": "inconclusive",
        "reason": "",
        "baseline_samples": (
            [_json_safe_sample(sample) for sample in baseline_original]
            if baseline_original is not None
            else None
        ),
        "candidate_samples": (
            [_json_safe_sample(sample) for sample in candidate_original]
            if candidate_original is not None
            else None
        ),
        "policy": {
            "ratio": ratio,
            "min_added_seconds": min_added_seconds,
            "max_relative_spread": max_relative_spread,
        },
        "summaries": {"baseline": None, "candidate": None},
        "ratio_of_medians": None,
        "median_added_seconds": None,
        "paired_regressions": None,
    }

    def finish(status: str, reason: str) -> dict[str, Any]:
        report["status"] = status
        report["reason"] = reason
        return report

    if baseline_original is None or candidate_original is None:
        return finish("inconclusive", "both sample arrays must be provided")
    if not baseline_original or not candidate_original:
        return finish("inconclusive", "both sample arrays must contain samples")
    if len(baseline_original) != len(candidate_original):
        return finish("inconclusive", "baseline and candidate counts differ")
    if len(baseline_original) < 5:
        return finish("inconclusive", "at least five paired samples are required")

    baseline_values = [_elapsed_seconds(sample) for sample in baseline_original]
    candidate_values = [_elapsed_seconds(sample) for sample in candidate_original]
    for side, values in (
        ("baseline", baseline_values),
        ("candidate", candidate_values),
    ):
        for index, value in enumerate(values):
            if value is None:
                return finish(
                    "inconclusive",
                    f"{side} sample at index {index} must be finite and positive",
                )

    valid_baseline = [value for value in baseline_values if value is not None]
    valid_candidate = [value for value in candidate_values if value is not None]
    baseline_summary = _summarize(valid_baseline)
    candidate_summary = _summarize(valid_candidate)
    report["summaries"] = {
        "baseline": baseline_summary,
        "candidate": candidate_summary,
    }

    baseline_median = baseline_summary["median_seconds"]
    candidate_median = candidate_summary["median_seconds"]
    report["ratio_of_medians"] = _finite_or_none(candidate_median / baseline_median)
    report["median_added_seconds"] = _finite_or_none(candidate_median - baseline_median)

    for summary in (baseline_summary, candidate_summary):
        noise_limit = max(
            max_relative_spread * summary["median_seconds"], min_added_seconds
        )
        if summary["middle_spread_seconds"] > noise_limit:
            return finish("inconclusive", "middle spread exceeds the noise limit")

    paired_regressions = sum(
        candidate_time > ratio * baseline_time
        and candidate_time - baseline_time > min_added_seconds
        for baseline_time, candidate_time in zip(valid_baseline, valid_candidate)
    )
    report["paired_regressions"] = paired_regressions
    median_regression = (
        candidate_median > ratio * baseline_median
        and candidate_median - baseline_median > min_added_seconds
    )
    required_pairs = (4 * len(valid_baseline) + 4) // 5

    if median_regression and paired_regressions >= required_pairs:
        return finish("regression", "median slowdown repeats across paired samples")
    if median_regression or paired_regressions >= 2:
        return finish("inconclusive", "slowdown was not repeatable or was borderline")
    return finish("passed", "no repeatable slowdown exceeded both thresholds")
