#!/usr/bin/env python3
"""Compare late edits and tool completion in a generated long OpenCode session."""

import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

from e2e_support import SCRATCH, require, sha256
from perf_history import enumerate_history
from perf_support import benchmark_report, measure_pairs


def prepare_source(source, original):
    shutil.copy2(source, original)
    with sqlite3.connect(original) as db:
        session, count = db.execute(
            "SELECT session_id,count(*) n FROM part GROUP BY session_id ORDER BY n DESC,session_id LIMIT 1"
        ).fetchone()
        for table in ("part", "message"):
            db.execute(f"DELETE FROM {table} WHERE session_id != ?", (session,))
        db.execute("DELETE FROM session WHERE id != ?", (session,))
        rows = list(db.execute("SELECT id,data FROM part ORDER BY time_created,id"))
        edit = next(
            (id, json.loads(data))
            for id, data in reversed(rows)
            if json.loads(data).get("type") == "text"
        )
        tool = next(
            (id, json.loads(data))
            for id, data in reversed(rows)
            if json.loads(data).get("type") == "tool"
        )
        completed = json.loads(json.dumps(tool[1]))
        completed["state"]["status"] = "completed"
        completed["state"]["output"] = (
            "completionbeacon Result: 17.\n" + "工具 output line\n" * 300
        )
        completed["state"].pop("error", None)
        tool[1]["state"]["status"] = "running"
        for field in ("output", "error"):
            tool[1]["state"].pop(field, None)
        db.execute(
            "UPDATE part SET data = ?, time_updated = time_updated + 1 WHERE id = ?",
            (json.dumps(tool[1]), tool[0]),
        )
        parts = {
            id: (session, json.loads(data))
            for id, data in db.execute("SELECT id,data FROM part")
        }
        messages = db.execute("SELECT count(*) FROM message").fetchone()[0]
    edit[1]["text"] += "\neditbeacon Corrected answer 日本語."
    return (
        parts,
        {"edit": edit, "completion": (tool[0], completed)},
        {"parts": count, "messages": messages, "session": session},
    )


def run(cases, report, original, parts, changes, repetitions, timeout):
    for case in cases.values():
        path = case.providers / "opencode.db"
        shutil.copy2(original, path)
        case.command(
            "import",
            "--provider",
            "opencode",
            "--path",
            path,
            "--progress",
            "none",
            timeout=timeout,
        )
        shutil.copytree(case.data, case.root / "prepared")

    for name, (part, data) in changes.items():

        def measure(case, repetition, *, name=name, part=part, data=data):
            path = case.providers / "opencode.db"
            shutil.copy2(original, path)
            shutil.rmtree(case.data)
            shutil.copytree(case.root / "prepared", case.data)
            case.command(
                "import",
                "--provider",
                "opencode",
                "--path",
                path,
                "--progress",
                "none",
                timeout=timeout,
            )
            with sqlite3.connect(path) as db:
                db.execute(
                    "UPDATE part SET data = ?,time_updated = time_updated + 10 WHERE id = ?",
                    (json.dumps(data), part),
                )
            result, sample = case.measured(
                name,
                "import",
                "--provider",
                "opencode",
                "--path",
                path,
                "--progress",
                "none",
                "--format",
                "json",
                timeout=timeout,
            )
            require(
                json.loads(result.stdout)["outcome"] == "completed", "import failed"
            )
            changed = {**parts, part: (parts[part][0], data)}
            actual = enumerate_history(case, changed, timeout)
            if repetition == -1:
                case.command(
                    "import",
                    "--provider",
                    "opencode",
                    "--path",
                    path,
                    "--full",
                    "--progress",
                    "none",
                    timeout=timeout,
                )
                require(
                    enumerate_history(case, changed, timeout) == actual,
                    "incremental differs from full import",
                )
            # Provider path differs between the two isolated data roots.
            normalized = json.dumps(actual, sort_keys=True).replace(str(path), "$INPUT")
            return normalized, sample

        report["workloads"][name] = record = {"part": part}
        measure_pairs(cases, repetitions, measure, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    require(args.repetitions > 0, "repetitions must be positive")
    scratch = SCRATCH / "assistant/tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="edit-pairs-", dir=scratch))
    source = args.corpus.resolve(strict=True)
    original = root / "original.db"
    binaries = {"before": args.baseline}
    if args.candidate:
        binaries["after"] = args.candidate
    with benchmark_report(
        root,
        binaries,
        argv=sys.argv,
        runner_sha256=sha256(Path(__file__)),
        source_sha256=sha256(source),
    ) as (cases, report):
        parts, changes, manifest = prepare_source(source, original)
        report.update(**manifest, input_sha256=sha256(original))
        run(cases, report, original, parts, changes, args.repetitions, args.timeout)
        require(sha256(source) == report["source_sha256"], "source corpus changed")


if __name__ == "__main__":
    main()
