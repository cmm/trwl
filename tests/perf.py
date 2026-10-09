#!/usr/bin/env python3
"""Measure CLI performance, or compare two supplied trwl executables."""

import argparse
import itertools
import json
import math
import os
import platform
import shutil
import signal
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True

from e2e_support import SCRATCH, require, sha256, write_json
from perf_compare import compare_samples
from perf_fixtures import Corpus, verify_corpus, verify_search, verify_show
from perf_semantic import QUERY_WORKLOADS, PreparedSemantic
from perf_support import PerformanceCase, summarize_resources, trial_order

WORKLOADS = ("initial-import", "unchanged-import", "append-import", "search", "show")


class Interrupted(BaseException):
    pass


def interrupted(signum, _frame):
    raise Interrupted(f"interrupted by signal {signum}")


def trial(
    root,
    binary,
    workspace,
    provider,
    shape,
    size,
    timeout,
    message_bytes=256,
):
    case = PerformanceCase(root, binary)
    report = {
        "result": "failed",
        "measurements": {},
        "stage": "fixture",
        "data_directory": str(case.data),
        "data_retained": True,
    }
    try:
        corpus = Corpus(
            case.providers / "input",
            workspace,
            provider,
            shape,
            size,
            message_bytes=message_bytes,
        )
        # Reading all known input bytes hashes the fixture and warms those pages.
        report["before"] = corpus.manifest()
        args = (
            "import",
            "--provider",
            provider,
            "--path",
            corpus.path,
            "--format",
            "json",
            "--progress",
            "none",
        )
        report["stage"] = "initial-import"
        _, report["measurements"]["initial-import"] = case.measured(
            "initial-import", *args, timeout=timeout
        )
        report["stage"] = "verify-initial"
        identities = verify_corpus(case, corpus)
        report["stage"] = "unchanged-import"
        _, report["measurements"]["unchanged-import"] = case.measured(
            "unchanged-import", *args, timeout=timeout
        )
        report["stage"] = "verify-unchanged"
        verify_corpus(case, corpus, identities)
        corpus.append()
        report["after"] = corpus.manifest()
        report["stage"] = "append-import"
        _, report["measurements"]["append-import"] = case.measured(
            "append-import", *args, timeout=timeout
        )
        report["stage"] = "verify-append"
        identities = verify_corpus(case, corpus, identities)
        search_args = (
            "search",
            corpus.query,
            "--workspace",
            str(corpus.workspace),
            "--backend",
            "lexical",
            "--refresh",
            "off",
            "--events",
            "--limit",
            "5",
            "--format",
            "json",
        )
        show_args = (
            "show",
            "event",
            identities[corpus.target_text][0],
            "--format",
            "json",
        )
        report["stage"] = "search"
        result, report["measurements"]["search"] = case.measured(
            "search", *search_args, timeout=timeout
        )
        verify_search(case, corpus, json.loads(result.stdout), identities)
        report["stage"] = "show"
        result, report["measurements"]["show"] = case.measured(
            "show", *show_args, timeout=timeout
        )
        verify_show(case, corpus, json.loads(result.stdout), identities)
        report.update(result="passed", stage="complete")
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
        if not isinstance(exc, Exception):
            raise
    finally:
        try:
            case.close()
        except Exception as exc:  # noqa: BLE001 - preserve cleanup failures in the report
            report.update(result="failed", cleanup_error=str(exc))
        write_json(root / "trial.json", report)
    return report


def summarize(report, args):
    if args.binary is not None:
        report["measurements"] = {
            f"{name}/{workload}": summarize_resources(
                [
                    pair["runs"]["binary"]["measurements"][workload]
                    for pair in scenario["pairs"]
                    if not pair["warmup"]
                    and "binary" in pair["runs"]
                    and pair["runs"]["binary"]["result"] == "passed"
                ]
            )
            for name, scenario in report["scenarios"].items()
            for workload in scenario["workloads"]
        }
        report["result"] = (
            "passed"
            if not report.get("error")
            and all(
                values["samples"] == args.repetitions
                for values in report["measurements"].values()
            )
            and all(
                run["result"] == "passed"
                for scenario in report["scenarios"].values()
                for pair in scenario["pairs"]
                for run in pair["runs"].values()
            )
            else "failed"
        )
        return
    comparisons = {}
    for name, scenario in report["scenarios"].items():
        for workload in scenario["workloads"]:
            samples = {side: [] for side in ("baseline", "candidate")}
            for pair in scenario["pairs"]:
                if pair["warmup"]:
                    continue
                for side, values in samples.items():
                    run = pair["runs"].get(side)
                    # Keep pair positions: an invalid baseline sample must not
                    # be paired with a candidate from a different repetition.
                    values.append(
                        run["measurements"][workload]["wall_seconds"]
                        if run is not None and run["result"] == "passed"
                        else None
                    )
            comparison = compare_samples(
                **samples,
                ratio=args.ratio,
                min_added_seconds=args.read_slack
                if workload in ("search", "show", "semantic-search", "hybrid-search")
                else args.import_slack,
                max_relative_spread=args.max_relative_spread,
            )
            comparisons[f"{name}/{workload}"] = comparison
    report["comparisons"] = comparisons
    # Size growth is descriptive, not an assertion of a particular complexity.
    report["size_growth"] = {}
    groups = {}
    for scenario in report["scenarios"].values():
        key = tuple(
            scenario[k]
            for k in (
                "profile",
                "provider",
                "shape",
                "message_bytes",
                "long_message_bytes",
            )
        )
        groups.setdefault(key, []).append(scenario)
    for key, scenarios in groups.items():
        if len(scenarios) < 2:
            continue
        scenarios.sort(key=lambda s: s["size"])
        for workload in scenarios[0]["workloads"]:
            report["size_growth"]["/".join(map(str, (*key, workload)))] = size_growth(
                scenarios, workload
            )
    failed = any(
        run["result"] != "passed"
        for s in report["scenarios"].values()
        for p in s["pairs"]
        for run in p["runs"].values()
    )
    statuses = {c["status"] for c in comparisons.values()}
    report["result"] = (
        "failed"
        if failed or report.get("error")
        else "regression"
        if "regression" in statuses
        else "inconclusive"
        if statuses != {"passed"}
        else "passed"
    )


def size_growth(scenarios, workload):
    values = {}
    for side in ("baseline", "candidate"):
        medians = []
        for scenario in (scenarios[0], scenarios[-1]):
            pairs = scenario["pairs"]
            times = [
                p["runs"][side]["measurements"][workload]["wall_seconds"]
                for p in pairs
                if not p["warmup"]
                and side in p["runs"]
                and p["runs"][side]["result"] == "passed"
            ]
            medians.append(statistics.median(times) if times else None)
        if all(value is not None and value > 0 for value in medians):
            values[side] = medians[1] / medians[0]
    return {"sizes": [scenarios[0]["size"], scenarios[-1]["size"]], **values}


def scenario_spec(
    profile, provider, shape, size, message_bytes, long_message_bytes=None
):
    return {
        "profile": profile,
        "provider": provider,
        "shape": shape,
        "size": size,
        "message_bytes": message_bytes,
        "long_message_bytes": long_message_bytes,
        "workloads": list(
            WORKLOADS
            if profile == "import"
            else ("semantic-index",)
            if profile == "semantic-index"
            else QUERY_WORKLOADS
        ),
    }


def scenario_specs(args):
    if args.profile == "reference":
        # Providers and session shapes exercise import, not the embedding executor.
        imports = [
            scenario_spec("import", p, s, n, width)
            for p, s, n, width in itertools.product(
                ("opencode", "codex"), ("long", "many"), (1000, 10000), (256, 4096)
            )
        ]
        return imports + [
            # 64 short documents cross four inference batches; mixed pairs add
            # multi-chunk documents and unequal token lengths within the corpus.
            scenario_spec("semantic-index", "opencode", "long", 128, 256),
            scenario_spec("semantic-index", "opencode", "long", 32, 256, 4096),
            scenario_spec("semantic-query", "opencode", "long", 100, 256),
            scenario_spec("semantic-query", "opencode", "long", 1000, 256),
        ]
    sizes = (
        args.sizes
        or (
            {
                "import": [1000, 10000],
                "semantic-index": [128],
                "semantic-query": [100, 1000],
            }[args.profile]
        )
    )
    providers = args.provider or (
        ["opencode", "codex"] if args.profile == "import" else ["opencode"]
    )
    shapes = args.shape or (["long", "many"] if args.profile == "import" else ["long"])
    return [
        scenario_spec(
            args.profile, p, s, n, args.message_bytes or 256, args.long_message_bytes
        )
        for p, s, n in itertools.product(
            dict.fromkeys(providers), dict.fromkeys(shapes), sizes
        )
    ]


def scenario_name(spec):
    name = "-".join(
        str(spec[k]) for k in ("profile", "provider", "shape", "size", "message_bytes")
    )
    return name + (
        f"-mixed-{spec['long_message_bytes']}" if spec["long_message_bytes"] else ""
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", type=Path, help="Measure one executable without comparisons"
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--baseline-label", help="Caller-supplied source/build label")
    parser.add_argument("--candidate-label", help="Caller-supplied source/build label")
    parser.add_argument("--binary-label", help="Caller-supplied source/build label")
    parser.add_argument(
        "--profile",
        choices=("import", "semantic-index", "semantic-query", "reference"),
        default="import",
        help="Reference runs the fixed import, embedding and query workloads",
    )
    parser.add_argument("--provider", action="append", choices=("opencode", "codex"))
    parser.add_argument("--shape", action="append", choices=("many", "long"))
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        metavar="MESSAGES",
        help="Message counts; defaults depend on the selected profile",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Model directory for semantic profiles; default: executable's supplied model",
    )
    parser.add_argument(
        "--message-bytes",
        type=int,
        help="ASCII message-body size; use larger bodies to measure content-heavy histories",
    )
    parser.add_argument(
        "--long-message-bytes",
        type=int,
        help="Alternate short/long user-reply pairs in semantic profiles",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Measured runs; comparisons need at least five",
    )
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--ratio", type=float, default=3.0)
    parser.add_argument(
        "--import-slack",
        type=float,
        default=0.25,
        help="Minimum added seconds for an import regression",
    )
    parser.add_argument(
        "--read-slack",
        type=float,
        default=0.05,
        help="Minimum added seconds for a read regression",
    )
    parser.add_argument("--max-relative-spread", type=float, default=0.5)
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Keep successful trials' data directories too",
    )
    args = parser.parse_args()
    if args.binary is not None:
        if args.baseline is not None or args.candidate is not None:
            parser.error("binary cannot be combined with baseline or candidate")
        sides = ("binary",)
    else:
        if args.baseline is None or args.candidate is None:
            parser.error("supply binary, or both baseline and candidate")
        sides = ("baseline", "candidate")
    if args.profile == "reference" and any(
        (
            args.provider,
            args.shape,
            args.sizes,
            args.message_bytes,
            args.long_message_bytes,
        )
    ):
        parser.error(
            "reference has fixed workloads; choose another profile to override them"
        )
    if args.profile != "import" and (args.provider or args.shape):
        parser.error("provider and shape selection apply only to import")
    if args.profile == "import" and (args.model_dir or args.long_message_bytes):
        parser.error("model-dir and long-message-bytes apply only to semantic profiles")
    if not (
        all(n > 0 for n in (args.sizes or []))
        and len(set(args.sizes or [])) == len(args.sizes or [])
        and (args.message_bytes is None or args.message_bytes >= 128)
        and (
            args.long_message_bytes is None
            or args.long_message_bytes > (args.message_bytes or 256)
        )
        and args.repetitions > 0
        and math.isfinite(args.timeout)
        and args.timeout > 0
    ):
        parser.error(
            "require distinct positive sizes, message bytes >= 128, longer mixed messages, positive repetitions and a finite positive timeout"
        )
    for slack in (args.import_slack, args.read_slack):
        try:
            compare_samples(
                [],
                [],
                ratio=args.ratio,
                min_added_seconds=slack,
                max_relative_spread=args.max_relative_spread,
            )
        except ValueError as exc:
            parser.error(str(exc))
    if sys.platform != "linux":
        parser.error(
            "process resource measurements currently require Linux wait4 semantics"
        )
    scratch = SCRATCH / "assistant/tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="trwl-perf-", dir=scratch))
    print(f"Artifacts: {root}", flush=True)
    workspace = root / "workspace"
    workspace.mkdir()
    specs = scenario_specs(args)
    report = {
        "argv": sys.argv,
        "root": str(root),
        "result": "failed",
        "binaries": {},
        "profile": args.profile,
        "model_dir": str(args.model_dir.resolve())
        if args.model_dir is not None
        else None,
        "repetitions": args.repetitions,
        "warmup_pairs_per_scenario": 1,
        "policy": {
            "ratio": args.ratio,
            "import_slack_seconds": args.import_slack,
            "read_slack_seconds": args.read_slack,
            "max_relative_spread": args.max_relative_spread,
        },
        "measurement": {
            "clock": "perf_counter, GNU time launch through blocking wait4 completion, including launcher overhead",
            "cache": "warm filesystem pages; fresh processes; one excluded warmup pair per scenario; semantic samples restore per-executable prepared states",
            "cpu": "wait4 user/system seconds for the command and waited descendants, plus detached inference worker /proc stat deltas at SC_CLK_TCK resolution",
            "memory": "GNU time %M command peak plus detached inference workers' lifetime VmHWM, in KiB; sum of peaks bounds concurrent RSS; worker snapshots and client_resources are separate",
            "io": "Linux wait4 block I/O (512-byte blocks) plus detached inference worker /proc io read_bytes/write_bytes deltas; cached reads are not block input; includes command output and GNU time's resource report",
            "faults": "wait4 minor/major page faults and voluntary/involuntary context switches",
        },
        "machine": {
            "platform": platform.platform(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
        "scenarios": {},
    }
    handlers = {
        sig: signal.signal(sig, interrupted)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    started = time.monotonic()
    prepared = {}
    try:
        for side in sides:
            binary = getattr(args, side).resolve(strict=True)
            require(
                binary.is_file() and os.access(binary, os.X_OK),
                f"{side} is not executable",
            )
            probe = PerformanceCase(root / f"{side}-identity", binary)
            try:
                version = probe.command("--version").stdout.strip()
            finally:
                probe.close()
            report["binaries"][side] = {
                "path": str(binary),
                "sha256": sha256(binary),
                "version": version,
                "source_build_label": getattr(args, side + "_label"),
            }
        for spec in specs:
            name = scenario_name(spec)
            scenario = {**spec, "pairs": [], "preparation": {}}
            report["scenarios"][name] = scenario
            scenario_root = root / name
            scenario_root.mkdir()
            prepared = {}
            if spec["profile"] != "import":
                for side in sides:
                    print(f"{name} {side}: preparing", flush=True)
                    prepared[side] = PreparedSemantic(
                        scenario_root / side,
                        Path(report["binaries"][side]["path"]),
                        workspace,
                        spec,
                        args.timeout,
                        args.model_dir.resolve() if args.model_dir else None,
                    )
                    scenario["preparation"][side] = prepared[side].preparation
                    write_json(root / "report.json", report)
            expected_manifests = None
            for repetition in range(-1, args.repetitions):
                pair = {
                    "warmup": repetition == -1,
                    "repetition": repetition,
                    "runs": {},
                }
                scenario["pairs"].append(pair)
                order = trial_order(sides, repetition)
                pair["order"] = order
                for side in order:
                    path = (
                        root / name / ("warmup" if repetition < 0 else str(repetition))
                    )
                    path.mkdir(parents=True, exist_ok=True)
                    if prepared:
                        (path / side).mkdir()
                        run = prepared[side].trial()
                    else:
                        run = trial(
                            path / side,
                            Path(report["binaries"][side]["path"]),
                            workspace,
                            spec["provider"],
                            spec["shape"],
                            spec["size"],
                            args.timeout,
                            spec["message_bytes"],
                        )
                    pair["runs"][side] = run
                    if run["result"] == "passed":
                        manifests = (
                            run["before"],
                            run["after"],
                        )
                        if expected_manifests is None:
                            expected_manifests = manifests
                        elif manifests != expected_manifests:
                            run.update(
                                result="failed",
                                error="fixture content differs between paired trials",
                            )
                    if (
                        not prepared
                        and run["result"] == "passed"
                        and not args.keep_data
                    ):
                        # Remove the entire disposable data root without inspecting
                        # its files, after correctness and fixture equality checks.
                        shutil.rmtree(run["data_directory"])
                        run["data_retained"] = False
                    write_json(path / side / "trial.json", run)
                    write_json(root / "report.json", report)
                    print(
                        f"{name} {'warmup' if repetition < 0 else repetition + 1} {side}: {run['result']}"
                        + (f" — {run['error']}" if "error" in run else ""),
                        flush=True,
                    )
                # Preserve the failure and continue other scenarios, without
                # running more repetitions of an already incorrect workload.
                if any(run["result"] != "passed" for run in pair["runs"].values()):
                    break
            keep = args.keep_data or any(
                run["result"] != "passed"
                for pair in scenario["pairs"]
                for run in pair["runs"].values()
            )
            for case in prepared.values():
                case.close(keep)
            scenario["prepared_data_retained"] = bool(prepared) and keep
            prepared = {}
        for side, identity in report["binaries"].items():
            identity["final_sha256"] = sha256(Path(identity["path"]))
            require(
                identity["final_sha256"] == identity["sha256"],
                f"{side} executable changed during the comparison",
            )
        summarize(report, args)
    except BaseException as exc:  # noqa: BLE001 - preserve cancellation and failure logs
        report.update(result="failed", error=str(exc), traceback=traceback.format_exc())
    finally:
        for case in prepared.values():
            case.close(True)
        report["elapsed_seconds"] = time.monotonic() - started
        report["exit_status"] = (
            0
            if report["result"] == "passed"
            else 2
            if report["result"] == "inconclusive"
            else 1
        )
        write_json(root / "report.json", report)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    for name, comparison in report.get("comparisons", {}).items():
        if comparison["status"] != "passed":
            print(
                f"{name}: {comparison['status']} — {comparison['reason']}", flush=True
            )
    if "error" in report:
        print(report["error"], flush=True)
    print(f"Result: {report['result']}; report: {root / 'report.json'}", flush=True)
    return report["exit_status"]


if __name__ == "__main__":
    raise SystemExit(main())
