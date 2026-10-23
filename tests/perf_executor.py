#!/usr/bin/env python3
"""Compare cold and reused CPU inference against a prepared semantic benchmark."""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

from e2e_support import SCRATCH, require, sha256
from perf_support import benchmark_report, inference_resources, measure_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    require(args.repetitions > 0, "repetitions must be positive")
    source = json.loads(args.prepared_report.read_text())
    scenarios = [
        s for s in source["scenarios"].values() if s["profile"] == "semantic-query"
    ]
    require(len(scenarios) == 1, "prepare exactly one semantic-query size")
    preparations = scenarios[0]["preparation"]
    require(len(preparations) == 1, "prepare with --binary to select one input")
    query_input = next(iter(preparations.values()))["query_input"]
    prepared = {name: Path(path) for name, path in query_input["directories"].items()}
    require(
        prepared["data"].is_dir(), "prepared data missing; prepare with --keep-data"
    )
    require(
        not inference_resources(prepared["runtime"]),
        "stop inference before copying prepared data",
    )
    scratch = SCRATCH / "assistant/tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="executor-pairs-", dir=scratch))
    with benchmark_report(
        root,
        {"before": args.baseline, "after": args.candidate},
        model_dir=args.model_dir.resolve(),
        argv=sys.argv,
        source_report_sha256=sha256(args.prepared_report),
        query_input=query_input,
        measurement="wait4 client resources plus detached inference /proc CPU and block I/O deltas; RSS is the sum of client peak and worker lifetime peak, an upper bound on concurrent RSS. Cold means no executor; filesystem caches are not evicted.",
    ) as (cases, report):
        for case in cases.values():
            for name in ("data", "config", "cache"):
                shutil.copytree(prepared[name], getattr(case, name), dirs_exist_ok=True)
        for name, backend, cold in [
            ("semantic-cold", "semantic", True),
            ("semantic-warm", "semantic", False),
            ("hybrid-warm", "hybrid", False),
        ]:
            argv = [
                "search",
                query_input["query"],
                "--workspace",
                query_input["workspace"],
                "--backend",
                backend,
                "--refresh",
                "off",
                "--events",
                "--limit",
                "3",
                "--format",
                "json",
            ]

            def measure(
                case, _repetition, *, cold=cold, argv=argv, name=name, backend=backend
            ):
                if cold:
                    case.command("daemon", "stop")
                else:
                    # The baseline can lack a reusable executor; its warm-up
                    # still exercises the same command and filesystem pages.
                    case.command(*argv)
                result, sample = case.measured(name, *argv)
                value = json.loads(result.stdout)
                require(
                    value["retrieval"]["effective_mode"] == backend
                    and value["retrieval"]["semantic_status"] == "ready"
                    and value["results"],
                    "prepared query did not retrieve semantic results",
                )
                return result.stdout, sample

            report["workloads"][name] = record = {}
            measure_pairs(cases, args.repetitions, measure, record)


if __name__ == "__main__":
    main()
