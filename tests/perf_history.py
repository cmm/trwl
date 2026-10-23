#!/usr/bin/env python3
"""Measure reads and imports of a frozen or generated OpenCode history."""

import argparse
import json
import math
import os
import platform
import re
import signal
import sqlite3
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

sys.dont_write_bytecode = True

from e2e_support import SCRATCH, require, sha256, write_json
from perf import interrupted
from perf_support import PerformanceCase, summarize_resources


def native_parts(path):
    """Read complete source parts."""
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        parts = {
            part: (session, json.loads(data))
            for part, session, data in db.execute("SELECT id,session_id,data FROM part")
        }
        manifest = {
            "input_sha256": sha256(path),
            "input_bytes": path.stat().st_size,
            "rows": {
                table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                for table in ("session", "message", "part")
            },
            "child_sessions": db.execute(
                "SELECT count(*) FROM session WHERE parent_id IS NOT NULL"
            ).fetchone()[0],
            "part_types": dict(Counter(data.get("type") for _, data in parts.values())),
        }
    require(parts, "history corpus has no parts")
    generated_manifest = path.with_name("manifest.json")
    if generated_manifest.exists():
        generated = json.loads(generated_manifest.read_text())
        if generated.get("corpus") == "synthetic-history-v1":
            require(
                generated["database_sha256"] == manifest["input_sha256"],
                "generated corpus hash differs",
            )
            manifest["generated"] = generated
    return parts, manifest


def enumerate_history(case, parts, timeout):
    """Verify source parts, full text/output, and pagination."""
    rows, cursors, cursor_args = [], set(), ()
    while True:
        result = case.command(
            "list",
            "events",
            "--workspace",
            "**",
            "--format",
            "jsonl",
            "--limit",
            "1000",
            *cursor_args,
            timeout=timeout,
        )
        page = [json.loads(line) for line in result.stdout.splitlines()]
        require(
            page and page[-1]["record_type"] == "event_range_completion",
            "list omitted completion",
        )
        require(
            all(row["record_type"] == "event_range_event" for row in page[:-1]),
            "unexpected list record",
        )
        rows.extend(row["event"] for row in page[:-1])
        completion = page[-1]
        cursor = completion.get("next_cursor")
        if completion["terminal"]:
            require(
                not completion["truncated"] and not cursor, "incomplete terminal page"
            )
            break
        require(
            completion["truncated"]
            and cursor
            and cursor not in cursors
            and len(page) > 1,
            "list cursor did not advance",
        )
        cursors.add(cursor)
        cursor_args = ("--cursor", cursor)
    require(
        len({row["trwl_event_id"] for row in rows}) == len(rows), "duplicate event IDs"
    )
    primary, outputs = {}, {}
    for row in rows:
        part = row["provider_event_id"]
        require(part in parts and row["provider"] == "opencode", "unknown native event")
        session, data = parts[part]
        require(row["provider_session_id"] == session, "wrong native session")
        if data.get("type") == "tool" and row["event_type"] == "tool_output":
            require(part not in outputs, "duplicate tool output")
            outputs[part] = row["text"]
        else:
            require(
                part not in primary and row["structured_content"] == data,
                "lost or duplicated native part content",
            )
            primary[part] = row
            if data.get("type") in ("text", "reasoning", "tool_result"):
                require(row["text"] == data.get("text", ""), "changed full native text")
    require(set(primary) == set(parts), "missing native parts")
    expected_outputs = {
        part: "\n".join(values)
        for part, (_, data) in parts.items()
        if data.get("type") == "tool"
        and (
            values := [
                data.get("state", {})[key]
                for key in ("output", "error")
                if isinstance(data.get("state", {}).get(key), str)
            ]
        )
    }
    require(outputs == expected_outputs, "changed, missing or extra full tool outputs")
    return rows


def trial(root, args, parts):
    case = PerformanceCase(root, args.binary, args.model_dir)
    # Default-workspace reads must run from the requested checkout, not the
    # runner's otherwise empty disposable workspace. Data and HOME stay private.
    case.work = args.workspace
    report = {"result": "failed", "measurements": {}, "checks": {}}
    try:
        report["version"] = case.command("--version").stdout.strip()
        case.command(
            "sources", "add", "history", "--provider", "opencode", "--root", args.corpus
        )

        def measure(name, *argv):
            result, record = case.measured(name, *argv, timeout=args.timeout)
            report["measurements"][name] = record
            return result

        result = measure(
            "initial-import",
            "import",
            "--all",
            "--progress",
            "none",
            "--format",
            "json",
        )
        require(
            json.loads(result.stdout)["outcome"] == "completed", "incomplete import"
        )
        rows = enumerate_history(case, parts, args.timeout)
        by_id = {row["trwl_event_id"]: row for row in rows}
        sessions = defaultdict(list)
        for row in rows:
            sessions[row["trwl_session_id"]].append(row)
        largest = max(sessions.values(), key=len)
        target = max(largest, key=lambda row: len(row["text"]))
        report["history"] = {
            "events": len(rows),
            "sessions": len(sessions),
            "event_types": dict(Counter(row["event_type"] for row in rows)),
            "workspaces": dict(Counter(row["workspace_root"] for row in rows)),
            "target_session_events": len(largest),
            "target_text_bytes": len(target["text"].encode()),
        }
        if args.generated is not None:
            require(
                None not in {row["workspace_root"] for row in rows}
                and len(report["history"]["workspaces"])
                == args.generated["parameters"]["workspaces"],
                "generated workspaces did not resolve separately",
            )
            require(
                sum(
                    events[0]["parent_trwl_session_id"] is not None
                    for events in sessions.values()
                )
                == args.generated["child_sessions"],
                "generated parent/child relationships changed",
            )
        result = measure(
            "unchanged-import",
            "import",
            "--all",
            "--progress",
            "none",
            "--format",
            "json",
        )
        require(
            all(
                source.get("unchanged") is True
                for source in json.loads(result.stdout)["sources"]
            ),
            "unchanged import reparsed the corpus",
        )
        require(
            enumerate_history(case, parts, args.timeout) == rows,
            "unchanged import changed content or identities",
        )

        def search(
            name,
            query,
            *filters,
            backend="lexical",
            refresh="off",
            events=True,
            expected_mode=None,
            empty=False,
            candidates=None,
            known_match=False,
        ):
            candidates = rows if candidates is None else candidates
            # A literal word in a selected source record establishes a match
            # without duplicating the CLI's query parser or ranking algorithm.
            if re.fullmatch(r"[a-zA-Z0-9]{1,40}", query):
                word = re.compile(r"\b" + re.escape(query) + r"\b", re.IGNORECASE)
                known_match |= any(word.search(row["text"]) for row in candidates)
            value = json.loads(
                measure(
                    name,
                    "search",
                    query,
                    "--backend",
                    backend,
                    "--refresh",
                    refresh,
                    "--limit",
                    "10",
                    "--format",
                    "json",
                    *(("--events",) if events else ()),
                    *filters,
                ).stdout
            )
            hits = value["results"]
            require(
                value["retrieval"]["effective_mode"] == (expected_mode or backend),
                "unexpected retrieval backend",
            )
            if backend != "lexical" and expected_mode is None:
                require(
                    value["retrieval"]["semantic_status"] == "ready",
                    "embeddings not ready",
                )
            require(
                len(hits) <= 10 and all(hit["trwl_event_id"] in by_id for hit in hits),
                "unknown or unbounded search results",
            )
            require(not empty or not hits, "no-match query returned results")
            require(
                not known_match or hits, f"{name}: search lost known source matches"
            )
            if backend != "lexical" and expected_mode is None:
                require(hits, f"{name}: ready semantic search returned no results")
            if not events:
                require(
                    len({hit["trwl_session_id"] for hit in hits}) == len(hits),
                    "grouped results repeated a session",
                )
            report["checks"][name] = {
                "query": query,
                "coverage": "matched" if hits else "empty",
                "known_match": known_match,
                "hits": len(hits),
                "sessions": len({hit["trwl_session_id"] for hit in hits}),
                "retrieval": value["retrieval"],
            }
            return hits

        for index, query in enumerate(args.query):
            for events, mode in ((True, "events"), (False, "sessions")):
                search(
                    f"lexical-{mode}-{index}", query, "--workspace", "**", events=events
                )
        phrase = next(
            (
                match.group()
                for row in rows
                if (
                    match := re.search(
                        r"\b[a-zA-Z]{2,20} [a-zA-Z]{2,20}\b", row["text"]
                    )
                )
            ),
            None,
        )
        if phrase is not None:
            search(
                "lexical-required",
                " ".join("+" + word for word in phrase.split()),
                "--workspace",
                "**",
                known_match=True,
            )
            search(
                "lexical-phrase", f'"{phrase}"', "--workspace", "**", known_match=True
            )
        else:
            for name in ("lexical-required", "lexical-phrase"):
                report["checks"][name] = {
                    "coverage": "skipped",
                    "reason": "no two-word source phrase",
                }
        search(
            "lexical-alternatives",
            args.query[0],
            "--term",
            args.query[-1],
            "--workspace",
            "**",
        )
        hits = search(
            "lexical-transcript",
            args.query[0],
            "--content-scope",
            "transcript",
            "--workspace",
            "**",
            candidates=[row for row in rows if row["event_type"] == "message"],
        )
        require(
            all(hit["event_type"] == "message" for hit in hits),
            "transcript filter leaked tools",
        )
        hits = search(
            "lexical-session",
            args.query[0],
            "--session",
            target["trwl_session_id"],
            "--workspace",
            "**",
            candidates=largest,
        )
        require(
            all(hit["trwl_session_id"] == target["trwl_session_id"] for hit in hits),
            "session filter leaked",
        )
        workspace_rows = [
            row for row in rows if row["workspace_root"] == str(args.workspace)
        ]
        search(
            "lexical-default-workspace",
            args.query[0],
            events=False,
            candidates=workspace_rows,
        )
        search(
            "lexical-workspace-glob",
            args.query[0],
            "--workspace",
            "**/" + args.workspace.name,
            events=False,
            candidates=workspace_rows,
        )
        target_workspace = target["workspace_root"]
        if target_workspace is not None:
            hits = search(
                "lexical-target-workspace",
                args.query[0],
                "--workspace",
                target_workspace,
                candidates=[
                    row for row in rows if row["workspace_root"] == target_workspace
                ],
            )
            require(
                all(
                    by_id[hit["trwl_event_id"]]["workspace_root"] == target_workspace
                    for hit in hits
                ),
                "workspace filter leaked",
            )
        hits = search(
            "lexical-primary",
            args.query[0],
            "--workspace",
            "**",
            "--primary-only",
            candidates=[row for row in rows if not row["is_subagent"]],
        )
        require(
            all(not by_id[hit["trwl_event_id"]]["is_subagent"] for hit in hits),
            "primary filter leaked children",
        )
        search(
            "lexical-miss", "trwlbenchnomatch7698321", "--workspace", "**", empty=True
        )
        search(
            "hybrid-disabled",
            args.query[0],
            "--workspace",
            "**",
            backend="hybrid",
            expected_mode="lexical",
        )

        event = json.loads(
            measure(
                "show-event",
                "show",
                "event",
                target["trwl_event_id"],
                "--format",
                "json",
            ).stdout
        )
        require(event["event"] == target, "show changed the complete target event")
        event = json.loads(
            measure(
                "show-window",
                "show",
                "event",
                target["trwl_event_id"],
                "--window",
                "5",
                "--format",
                "json",
            ).stdout
        )
        position = largest.index(target)
        require(
            event["events"] == largest[max(0, position - 5) : position + 6],
            "wrong full event window",
        )
        session = json.loads(
            measure(
                "show-session",
                "show",
                "session",
                target["trwl_session_id"],
                "--format",
                "json",
            ).stdout
        )
        require(session["events"] == largest, "show lost full session records")
        markdown = measure(
            "show-session-markdown",
            "show",
            "session",
            target["trwl_session_id"],
            "--format",
            "markdown",
        )
        require(target["text"] in markdown.stdout, "Markdown lost full target text")
        location = json.loads(
            measure(
                "locate-event",
                "locate",
                "event",
                target["trwl_event_id"],
                "--format",
                "json",
            ).stdout
        )
        require(
            location["trwl_event_id"] == target["trwl_event_id"],
            "locate changed identity",
        )
        cursor_args = ()
        offset = 0
        while True:
            name = (
                "list-first-page"
                if not cursor_args
                else f"list-page-{offset // 1000 + 1}"
            )
            output = measure(
                name,
                "list",
                "events",
                "--workspace",
                "**",
                "--limit",
                "1000",
                "--format",
                "jsonl",
                *cursor_args,
            )
            page = [json.loads(line) for line in output.stdout.splitlines()]
            events = [row["event"] for row in page[:-1]]
            require(
                events == rows[offset : offset + len(events)],
                "list page changed records",
            )
            offset += len(events)
            cursor = page[-1].get("next_cursor")
            if not cursor:
                require(
                    offset == len(rows) and page[-1]["terminal"],
                    "incomplete measured list",
                )
                break
            require(events and offset < len(rows), "measured list did not advance")
            cursor_args = ("--cursor", cursor)

        if args.semantic:
            config = case.config / "config.toml"
            config.write_text(
                config.read_text() + "[semantic]\nbuiltin_throttling = false\n"
            )
            case.command(
                "semantic", "enable", "--executor", "builtin", timeout=args.timeout
            )
            topic = "Why did workspace filtering miss session history?"
            search(
                "embedding-fill",
                topic,
                "--workspace",
                "**",
                backend="semantic",
                refresh="wait",
            )
            report["semantic_coverage"] = case.json_command("semantic", "status")
            for backend in ("semantic", "hybrid"):
                for refresh in ("off", "wait"):
                    for events, mode in ((True, "events"), (False, "sessions")):
                        search(
                            f"{backend}-{refresh}-{mode}",
                            topic,
                            "--workspace",
                            "**",
                            backend=backend,
                            refresh=refresh,
                            events=events,
                        )
        report["result"] = "passed"
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
        if not isinstance(exc, Exception):
            raise
    finally:
        case.close()
        write_json(root / "trial.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Frozen or generated OpenCode SQLite database",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Checkout for default-workspace queries",
    )
    parser.add_argument(
        "--query",
        action="append",
        help="Broad lexical query; default: nix, benchmark, workspace",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Also measure CPU embedding fill and ready queries",
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.model_dir is not None and not args.semantic:
        parser.error("model-dir requires semantic")
    if any(not query.strip() for query in args.query or []):
        parser.error("queries must not be empty")
    if (
        sys.platform != "linux"
        or args.repetitions < 1
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        parser.error(
            "require Linux, positive repetitions, and a finite positive timeout"
        )
    for name in ("binary", "corpus", "workspace", "model_dir"):
        if (path := getattr(args, name)) is not None:
            setattr(args, name, path.resolve(strict=True))
    args.query = args.query or ["nix", "benchmark", "workspace"]
    require(args.workspace.is_dir(), "workspace must be a directory")
    parts, manifest = native_parts(args.corpus)
    args.generated = manifest.get("generated")
    scratch = SCRATCH / "assistant/tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="trwl-perf-history-", dir=scratch))
    print(f"Artifacts: {root}", flush=True)
    report = {
        "argv": sys.argv,
        "result": "failed",
        "corpus": manifest,
        "binary": {"path": str(args.binary), "sha256": sha256(args.binary)},
        "model_files": {
            name: {
                "bytes": (args.model_dir / name).stat().st_size,
                "sha256": sha256(args.model_dir / name),
            }
            for name in (
                "onnx/model.onnx",
                "tokenizer.json",
                "config.json",
                "special_tokens_map.json",
                "tokenizer_config.json",
            )
        }
        if args.model_dir is not None
        else None,
        "machine": {
            "platform": platform.platform(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
        "measurement": {
            "cache": "warm provider pages and host caches; fresh CLI process per operation; one excluded warmup trial",
            "wall": "GNU time launch through wait4 completion, including launcher overhead",
            "cpu": "wait4 command/descendant CPU plus detached inference worker /proc stat deltas at SC_CLK_TCK resolution",
            "memory": "GNU time command peak plus detached inference workers' lifetime VmHWM in KiB; sum of peaks bounds concurrent RSS; client and worker snapshots are separate",
            "io": "wait4 filesystem I/O (512-byte blocks) plus detached inference worker /proc io deltas; cached reads are not block input; command output and GNU time's resource report are included",
            "semantic": "foreground CPU inference, unthrottled; empty embeddings at each trial; preparation and ready reads are separate",
        },
        "trials": [],
    }
    handlers = {
        sig: signal.signal(sig, interrupted)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    started = time.monotonic()
    try:
        for repetition in range(-1, args.repetitions):
            run = trial(
                root / ("warmup" if repetition < 0 else str(repetition)), args, parts
            )
            report["trials"].append({"warmup": repetition < 0, **run})
            write_json(root / "report.json", report)
            print(
                f"{'warmup' if repetition < 0 else repetition + 1}: {run['result']}",
                flush=True,
            )
            if run["result"] != "passed":
                break
        report["binary"]["final_sha256"] = sha256(args.binary)
        require(
            report["binary"]["final_sha256"] == report["binary"]["sha256"],
            "executable changed during run",
        )
        require(
            sha256(args.corpus) == manifest["input_sha256"],
            "corpus changed during run; use a frozen snapshot",
        )
        measured = [
            run
            for run in report["trials"]
            if not run["warmup"] and run["result"] == "passed"
        ]
        workloads = {name for run in report["trials"] for name in run["measurements"]}
        report["measurements"] = {
            name: summarize_resources(
                [
                    run["measurements"][name]
                    for run in measured
                    if name in run["measurements"]
                ]
            )
            for name in sorted(workloads)
        }
        report["result"] = (
            "passed"
            if len(measured) == args.repetitions
            and all(run["result"] == "passed" for run in report["trials"])
            else "failed"
        )
    except BaseException as exc:  # noqa: BLE001 - preserve cancellation and failure logs
        report.update(error=str(exc), traceback=traceback.format_exc())
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(root / "report.json", report)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    print(f"Result: {report['result']}; report: {root / 'report.json'}", flush=True)
    return 0 if report["result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
