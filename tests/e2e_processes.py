"""Refresh, interruption recovery, and overlapping imports through the CLI."""

import json
import os
import shutil
import signal
import time
from collections import Counter
from contextlib import ExitStack

from e2e_support import OpenCodeFixture, codex_fixture, require


def _message(case, marker, text, provider, provider_session, *, identity=None):
    hits = case.search(marker, "--events", "--limit", "200")["results"]
    require(len(hits) == 1, f"expected one {marker} hit, got {len(hits)}")
    hit = hits[0]
    actual_identity = (hit["trwl_event_id"], hit["trwl_session_id"])
    if identity is not None:
        require(actual_identity == identity, f"{marker} IDs changed")
    event = case.json_command("show", "event", hit["trwl_event_id"])["event"]
    require(
        (event["trwl_event_id"], event["trwl_session_id"]) == actual_identity
        and event["provider"] == provider
        and event["provider_session_id"] == provider_session
        and event["text"] == text,
        f"show lost {marker} identity or complete text",
    )
    return actual_identity


def _listed_messages(case, session_id, texts):
    """Check complete messages and unique IDs across all result pages."""
    deadline = time.monotonic() + 120
    rows = []
    cursors = set()
    cursor_args = ()
    while True:
        remaining = deadline - time.monotonic()
        require(remaining > 0, f"listing session {session_id} timed out")
        page = case.json_command(
            "list",
            "events",
            "--session",
            session_id,
            "--event-type",
            "message",
            "--limit",
            "1000",
            *cursor_args,
            timeout=min(30, remaining),
        )
        rows.extend(page["events"])
        cursor = page.get("next_cursor")
        if not cursor:
            break
        require(cursor not in cursors, "list repeated its continuation cursor")
        cursors.add(cursor)
        cursor_args = ("--cursor", cursor)
    require(
        Counter(row["text"] for row in rows) == Counter(texts),
        f"session {session_id} lost messages, changed full text, or has duplicates",
    )
    require(
        len({row["trwl_event_id"] for row in rows}) == len(rows)
        and all(row["trwl_session_id"] == session_id for row in rows),
        f"session {session_id} has repeated event IDs or unrelated events",
    )


def refresh_restart(case):
    """Check refresh modes, updater restarts and a copied data directory."""
    session = "ses_process_refresh"
    fixture = OpenCodeFixture(case.providers / "refresh.db", case.work)
    fixture.session(session)
    old = "restartoldbeacon Original history.\nKeep this complete second line."
    waited = "restartwaitbeacon Appended history visible after waited refresh."
    background = (
        "restartbackgroundbeacon Automatic refresh preserves this full message."
    )
    restarted = (
        "restartdaemonbeacon History written while the background process was stopped."
    )
    fixture.message(session, "msg_old", old)
    # Register a refreshable root explicitly rather than depending on host discovery.
    case.command(
        "sources",
        "add",
        "process-refresh",
        "--provider",
        "opencode",
        "--root",
        fixture.path,
    )
    case.command("import", "--all")
    old_id = _message(case, "restartoldbeacon", old, "opencode", session)

    fixture.message(session, "msg_wait", waited)
    require(
        not case.search("restartwaitbeacon", "--events", refresh="off")["results"],
        "refresh off saw an unimported append",
    )
    _message(case, "restartoldbeacon", old, "opencode", session, identity=old_id)
    require(
        case.search("restartwaitbeacon", "--events", refresh="wait")["results"],
        "waited refresh missed the append",
    )
    waited_id = _message(case, "restartwaitbeacon", waited, "opencode", session)
    _message(case, "restartoldbeacon", old, "opencode", session, identity=old_id)
    case.check(
        "refresh off excludes append; waited refresh preserves old IDs/full text"
    )

    # Writing the documented mode avoids the mode setter's daemon autostart. The
    # only persistent worker we launch directly is an owned foreground process.
    config = case.config / "config.toml"
    manual_config = config.read_text()
    config.write_text(manual_config.replace('mode = "manual"', 'mode = "auto"', 1))
    daemon = None
    try:
        daemon = case.start("daemon", "run", "--loop-interval-seconds", "1")
        deadline = time.monotonic() + 20
        while True:
            require(daemon.process.poll() is None, f"daemon exited: {daemon.log}")
            remaining = deadline - time.monotonic()
            require(remaining > 0, f"daemon readiness timed out: {daemon.log}")
            status = case.json_command("status", timeout=min(5, remaining))
            if status.get("daemon", {}).get("running"):
                break
            time.sleep(0.05)

        fixture.message(session, "msg_background", background)
        require(daemon.process.poll() is None, f"daemon exited: {daemon.log}")
        case.search(
            "restartbackgroundbeacon", "--events", refresh="background", timeout=10
        )
        deadline = time.monotonic() + 45
        while True:
            require(daemon.process.poll() is None, f"daemon exited: {daemon.log}")
            remaining = deadline - time.monotonic()
            require(
                remaining > 0, "background refresh did not expose the appended content"
            )
            result = case.search(
                "restartbackgroundbeacon",
                "--events",
                refresh="off",
                timeout=min(5, remaining),
            )
            if result["results"]:
                break
            time.sleep(0.1)
        background_id = _message(
            case, "restartbackgroundbeacon", background, "opencode", session
        )
        _message(case, "restartoldbeacon", old, "opencode", session, identity=old_id)
        case.check(
            "automatic background refresh eventually exposes complete appended content"
        )
        before_stop = config.read_bytes()
        case.command("daemon", "stop", timeout=20)
        require(config.read_bytes() == before_stop, "daemon stop rewrote configuration")
        require(
            not case.json_command("daemon", "status")["running"],
            "daemon stop left the updater running",
        )
        require(
            case.json_command("index", "mode")["mode"] == "auto",
            "daemon stop changed indexing mode",
        )
        case.wait(daemon)
        fixture.message(session, "msg_restarted", restarted)
        daemon = case.start("daemon", "run", "--loop-interval-seconds", "1")
        deadline = time.monotonic() + 45
        while True:
            require(
                daemon.process.poll() is None, f"restarted daemon exited: {daemon.log}"
            )
            remaining = deadline - time.monotonic()
            require(remaining > 0, "restarted daemon did not index the new message")
            if case.search(
                "restartdaemonbeacon",
                "--events",
                refresh="off",
                timeout=min(5, remaining),
            )["results"]:
                break
            time.sleep(0.1)
        restarted_id = _message(
            case, "restartdaemonbeacon", restarted, "opencode", session
        )
        _message(case, "restartoldbeacon", old, "opencode", session, identity=old_id)
        case.check(
            "a restarted background process discovers new history and preserves existing references"
        )
    finally:
        # The public mode command also stops a replacement daemon if background
        # search started one after our foreground worker unexpectedly exited.
        # This root belongs only to this case. Always reap the owned group too.
        try:
            case.command("index", "mode", "manual", timeout=20)
        finally:
            if daemon is not None:
                case.stop(daemon)

    records = [
        ("restartoldbeacon", old, old_id),
        ("restartwaitbeacon", waited, waited_id),
        ("restartbackgroundbeacon", background, background_id),
        ("restartdaemonbeacon", restarted, restarted_id),
    ]
    # Each retrieval is a fresh executable invocation, including after shutdown.
    for marker, text, identity in records:
        _message(case, marker, text, "opencode", session, identity=identity)
    _listed_messages(case, old_id[1], [old, waited, background, restarted])
    case.check(
        "fresh processes after daemon shutdown preserve event/session IDs and full text"
    )

    copied = case.root / "reopened-data"
    shutil.copytree(case.data, copied / "trwl")
    previous_root = case.env["XDG_DATA_HOME"]
    case.env["XDG_DATA_HOME"] = str(copied)
    try:
        for marker, text, identity in records:
            _message(case, marker, text, "opencode", session, identity=identity)
        _listed_messages(case, old_id[1], [old, waited, background, restarted])
    finally:
        case.env["XDG_DATA_HOME"] = previous_root
    case.check(
        "copied data directory reopens the same IDs and full text without import"
    )


def interruption(case):
    """Interrupt one real Codex JSONL import after emitted progress, then retry."""
    fixture = OpenCodeFixture(case.providers / "prior.db", case.work)
    session = "ses_process_interruption_prior"
    fixture.session(session)
    old = (
        "interruptpriorbeacon Previously imported history.\nIt must survive the signal."
    )
    fixture.message(session, "msg_prior", old)
    case.command("import", "--provider", "opencode", "--path", fixture.path)
    old_id = _message(case, "interruptpriorbeacon", old, "opencode", session)

    # Keep the import running long enough to interrupt after its progress message.
    codex_session = "550e8400-e29b-41d4-a716-446655440201"
    texts = [
        f"interruptbatchbeacon{i:05d} Provider message {i}. "
        + "Preserve this native transcript content. " * 12
        for i in range(6000)
    ]
    path = codex_fixture(
        case.providers / "interrupt.jsonl",
        case.work,
        codex_session,
        [("user", text) for text in texts],
    )
    command = case.start(
        "import",
        "--provider",
        "codex",
        "--path",
        path,
        "--format",
        "json",
        "--progress",
        "json",
        timeout=30,
    )
    try:
        deadline = time.monotonic() + 15
        while True:
            require(
                command.process.poll() is None,
                f"import exited before interruption was established: {command.log}",
            )
            # Only public progress output establishes executable startup. Do not
            # select a phase or assert where the implementation was interrupted.
            progress = False
            for line in command.stderr_path.read_text(errors="replace").splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict) and item.get("type") == "trwl_progress":
                    progress = True
                    break
            if progress:
                break
            require(
                time.monotonic() < deadline,
                f"import emitted no progress: {command.log}",
            )
            time.sleep(0.005)
        require(
            command.process.poll() is None, f"import already finished: {command.log}"
        )
        os.killpg(command.process.pid, signal.SIGINT)
        result = case.wait(command, expect_failure=True)
        require(
            result.returncode in (130, -signal.SIGINT),
            f"import did not exit from SIGINT: {command.log}",
        )
    finally:
        case.stop(command)
    _message(case, "interruptpriorbeacon", old, "opencode", session, identity=old_id)
    case.check("history remains usable after SIGINT interrupts an import")

    case.command("import", "--provider", "codex", "--path", path, timeout=120)
    selected = None
    for i in (0, len(texts) // 2, len(texts) - 1):
        identity = _message(
            case, f"interruptbatchbeacon{i:05d}", texts[i], "codex", codex_session
        )
        if selected is None:
            selected = identity
        require(identity[1] == selected[1], "retry split the Codex session")
    _listed_messages(case, selected[1], texts)
    _message(case, "interruptpriorbeacon", old, "opencode", session, identity=old_id)
    case.check(
        "retry imports all expected Codex messages once and preserves prior history"
    )


def overlap(case):
    """Concurrently import OpenCode SQLite and Codex JSONL without input changes."""
    fixture = OpenCodeFixture(case.providers / "overlap.db", case.work)
    session = "ses_process_overlap"
    fixture.session(session)
    old = "overlappriorbeacon Original OpenCode message.\nPreserve its identity."
    added = "overlapappendbeacon OpenCode append present before either process starts."
    fixture.message(session, "msg_prior", old)
    case.command("import", "--provider", "opencode", "--path", fixture.path)
    old_id = _message(case, "overlappriorbeacon", old, "opencode", session)
    fixture.message(session, "msg_append", added)

    codex_session = "550e8400-e29b-41d4-a716-446655440202"
    texts = [
        f"overlapcodexbeacon{i:05d} Independent Codex message {i}. "
        + "Keep concurrent history complete. " * 12
        for i in range(512)
    ]
    path = codex_fixture(
        case.providers / "overlap.jsonl",
        case.work,
        codex_session,
        [("user", text) for text in texts],
    )
    appended_id = None
    codex_ids = {}
    # Both inputs are frozen before launch. The expected union does not depend on
    # which updater reads or publishes first, including on the unchanged repeat.
    for attempt in range(2):
        with ExitStack() as cleanup:
            first = case.start(
                "import", "--provider", "codex", "--path", path, timeout=120
            )
            cleanup.callback(case.stop, first)
            second = case.start(
                "import", "--provider", "opencode", "--path", fixture.path, timeout=120
            )
            cleanup.callback(case.stop, second)
            require(
                first.process.poll() is None,
                f"first import completed before process overlap was established: {first.log}",
            )
            case.wait(first)
            case.wait(second)

        _message(case, "overlappriorbeacon", old, "opencode", session, identity=old_id)
        appended_id = _message(
            case,
            "overlapappendbeacon",
            added,
            "opencode",
            session,
            identity=appended_id,
        )
        require(appended_id[1] == old_id[1], "overlap split the OpenCode session")
        for i in (0, len(texts) - 1):
            codex_ids[i] = _message(
                case,
                f"overlapcodexbeacon{i:05d}",
                texts[i],
                "codex",
                codex_session,
                identity=codex_ids.get(i),
            )
        require(
            codex_ids[0][1] == codex_ids[len(texts) - 1][1],
            "overlap split Codex history",
        )
        _listed_messages(case, old_id[1], [old, added])
        _listed_messages(case, codex_ids[0][1], texts)
        case.check(
            f"overlapping import pass {attempt + 1}: both successful histories present once; IDs/full text preserved"
        )
