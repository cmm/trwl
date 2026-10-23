#!/usr/bin/env python3
"""Test trwl's CLI workflows against an explicitly supplied executable."""

import argparse
import os
import signal
import sys
import tempfile
import traceback
from pathlib import Path

sys.dont_write_bytecode = True

from e2e_history import bad_input, selection, updates
from e2e_http import hybrid_grouping, semantic_cache, semantic_http
from e2e_live import busy_source, many_inputs, semantic_live, semantic_recovery
from e2e_opencode import opencode_v2
from e2e_processes import interruption, overlap, refresh_restart
from e2e_providers import provider_cases
from e2e_retrieval import retrieval
from e2e_semantic import (
    background_semantic,
    device_policy,
    executor_reuse,
    passive_semantic,
    semantic,
    unavailable,
)
from e2e_storage import index_storage
from e2e_support import SCRATCH, Case, MissingPrerequisite, require, sha256, write_json
from e2e_workflows import (
    benchmark_restore,
    discovery,
    file_search,
    jsonl_updates,
    long_content,
    xdg_storage,
)
from e2e_workspaces import workspaces

CASES = {
    "xdg-storage": xdg_storage,
    "hybrid-grouping": hybrid_grouping,
    "semantic-cache": semantic_cache,
    "benchmark-restore": benchmark_restore,
    "discovery": discovery,
    "retrieval": retrieval,
    "opencode-v2": opencode_v2,
    "selection": selection,
    "workspaces": workspaces,
    "updates": updates,
    "index-storage": index_storage,
    "busy-source": busy_source,
    "many-inputs": many_inputs,
    "jsonl-updates": jsonl_updates,
    "file-search": file_search,
    "long-content": long_content,
    "bad-input": bad_input,
    "refresh-restart": refresh_restart,
    "interruption": interruption,
    "overlap": overlap,
    **provider_cases,
    "semantic-unavailable": unavailable,
    "semantic": semantic,
    "semantic-passive": passive_semantic,
    "semantic-background": background_semantic,
    "semantic-device-policy": device_policy,
    "semantic-executor": executor_reuse,
    "semantic-http": semantic_http,
    "semantic-live": semantic_live,
    "semantic-recovery": semantic_recovery,
}


class Interrupted(BaseException):
    pass


def interrupted(signum, _frame):
    # InterruptedError can be retried by subprocess; unwind instead.
    raise Interrupted(f"interrupted by signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        required=True,
        help="Absolute or relative path to the trwl executable",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=CASES,
        help="Repeat to select cases; default: all registered cases",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Real model directory (default: executable's supplied model)",
    )
    parser.add_argument(
        "--source-revision", help="Caller-supplied label; no provenance is inferred"
    )
    args = parser.parse_args()
    selected = list(dict.fromkeys(args.case or CASES))
    SCRATCH.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="trwl-e2e-", dir=SCRATCH))
    print(f"Artifacts: {root}", flush=True)
    report = {
        "argv": sys.argv,
        "binary_argument": str(args.binary),
        "source_revision": args.source_revision,
        "selected_cases": selected,
        "cases": {},
        "coverage": "selected cases only; representative provider formats, not every historical schema",
        "result": "failed",
    }
    for name in CASES:
        report["cases"][name] = {
            "result": "not_run" if name in selected else "not_selected"
        }
    handlers = {
        sig: signal.signal(sig, interrupted)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        binary = args.binary.resolve(strict=True)
        require(
            binary.is_file() and os.access(binary, os.X_OK),
            f"--binary must name an executable file: {args.binary}",
        )
        report.update(
            original_binary=str(args.binary.absolute()),
            resolved_binary=str(binary),
            binary_sha256=sha256(binary),
        )
        model_dir = args.model_dir.resolve() if args.model_dir is not None else None
        report["model_dir"] = str(model_dir) if model_dir is not None else None
        for name in selected:
            case = None
            result = report["cases"][name]
            try:
                case = Case(root / name, binary, model_dir)
                result.update(
                    artifact_directory=str(case.root),
                    invoked_binary=str(case.binary),
                    binary_sha256=sha256(case.binary),
                )
                result["version"] = case.command("--version").stdout.strip()
                CASES[name](case)
                result["result"] = "passed"
            except MissingPrerequisite as exc:
                result.update(result="unrun", missing_prerequisite=str(exc))
            except BaseException as exc:
                result.update(
                    result="failed", error=str(exc), traceback=traceback.format_exc()
                )
                if not isinstance(exc, Exception):
                    raise
            finally:
                if case is not None:
                    try:
                        case.close()
                    except Exception as exc:  # noqa: BLE001 - report cleanup failures
                        result.update(
                            result="failed",
                            cleanup_error=str(exc),
                            cleanup_traceback=traceback.format_exc(),
                        )
                    result["checks"] = case.checks
                write_json(root / f"{name}-result.json", result)
        outcomes = {report["cases"][name]["result"] for name in selected}
        report["result"] = (
            "selected_cases_passed"
            if outcomes == {"passed"}
            else "incomplete"
            if outcomes <= {"passed", "unrun"}
            else "failed"
        )
    except BaseException as exc:  # noqa: BLE001 - record errors and signals in artifacts
        report.update(error=str(exc), traceback=traceback.format_exc())
    finally:
        report["exit_status"] = 0 if report["result"] == "selected_cases_passed" else 1
        write_json(root / "report.json", report)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        for name, result in report["cases"].items():
            print(
                f"{name}: {result['result']}"
                + (f" — {result['error']}" if "error" in result else ""),
                flush=True,
            )
            if "missing_prerequisite" in result:
                print(
                    f"  Missing prerequisite: {result['missing_prerequisite']}",
                    flush=True,
                )
        if "error" in report:
            print(f"Runner failed: {report['error']}", flush=True)
        if set(selected) != set(CASES):
            print("Partial run: unselected cases are not covered.", flush=True)
        print(f"Report: {root / 'report.json'}", flush=True)
    return report["exit_status"]


if __name__ == "__main__":
    raise SystemExit(main())
