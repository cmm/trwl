"""CLI selection, updates, and bad-input cases for OpenCode SQLite/Codex JSONL."""

import json
import re
import shutil
import sqlite3
from pathlib import Path

from e2e_support import OpenCodeFixture, codex_fixture, git, require


def _shown(case, hits):
    events = {}
    for hit in hits:
        event_id = hit["trwl_event_id"]
        require(event_id not in events, f"duplicate search event {event_id}")
        event = case.json_command("show", "event", event_id)["event"]
        require(
            event["trwl_event_id"] == event_id
            and event["trwl_session_id"] == hit["trwl_session_id"],
            f"search/show identity differs for {event_id}",
        )
        events[event_id] = event
    return events


def _events(case, query, *args, all_workspaces=True):
    if all_workspaces:
        args = ("--workspace", "**", *args)
    return _shown(
        case, case.search(query, "--events", "--limit", "100", *args)["results"]
    )


def _same_event(actual, expected):
    for field in (
        "trwl_event_id",
        "trwl_session_id",
        "provider",
        "provider_session_id",
        "event_type",
        "text",
    ):
        require(
            actual[field] == expected[field],
            f"changed {field} for {expected['trwl_event_id']}: {actual[field]!r}",
        )


def _message(case, marker, text, provider, session):
    events = _events(case, marker)
    require(len(events) == 1, f"expected one {marker} message, got {len(events)}")
    event = next(iter(events.values()))
    require(
        event["text"] == text
        and event["provider"] == provider
        and event["provider_session_id"] == session
        and event["event_type"] == "message",
        f"wrong full message or provider identity for {marker}: {event!r}",
    )
    return event


def selection(case):
    marker = "selectionorchard"
    parent, child, other = (
        "ses_selection_parent",
        "ses_selection_child",
        "ses_selection_other",
    )
    current = "550e8400-e29b-41d4-a716-446655440010"
    peer = "550e8400-e29b-41d4-a716-446655440011"
    other_work = case.root / "other-workspace"
    other_work.mkdir()
    git(case.work, "init", "--quiet")
    git(other_work, "init", "--quiet")
    fixture = OpenCodeFixture(case.providers / "selection.db", case.work)
    fixture.session(parent)
    fixture.session(child, parent=parent, agent="explore")
    fixture.session(other)
    bodies = {
        "parent-a": (
            "opencode",
            parent,
            f"{marker} Parent request in the recent workspace.",
        ),
        "parent-b": (
            "opencode",
            parent,
            f"{marker} A second complete parent reply.\nKeep this line.",
        ),
        "child": (
            "opencode",
            child,
            f"{marker} Older child conversation in the same workspace.",
        ),
        "other": (
            "opencode",
            other,
            f"{marker} Recent independent root in another workspace.",
        ),
        "current": ("codex", current, f"{marker} Current Codex conversation."),
        "peer": ("codex", peer, f"{marker} Another Codex conversation, not current."),
    }
    for name in ("parent-a", "parent-b", "child", "other"):
        _, session, text = bodies[name]
        fixture.message(session, "msg_" + name, text, role="assistant")
    tool_input = {"command": f"printf '{marker} call input'"}
    tool_output = f"{marker} Complete tool output.\nResult: 17."
    fixture.part(
        parent,
        "msg_parent-b",
        "part_selection_call",
        {
            "type": "tool",
            "callID": "call_selection",
            "tool": "bash",
            "state": {
                "status": "completed",
                "input": tool_input,
                "output": tool_output,
            },
        },
    )
    fixture.part(
        parent,
        "msg_parent-b",
        "part_selection_output",
        {
            "type": "tool_result",
            "callID": "call_selection",
            "text": tool_output,
        },
    )
    # Provider timestamps put real matching records on both sides of --since.
    with sqlite3.connect(fixture.path) as db:
        db.execute(
            "UPDATE session SET directory = ? WHERE id = ?", (str(other_work), other)
        )
        for session, stamp in (
            (parent, 1790035200000),
            (child, 1789862400000),
            (other, 1790121600000),
        ):
            db.execute(
                "UPDATE session SET time_created = ?, time_updated = ? WHERE id = ?",
                (stamp, stamp, session),
            )
            for table in ("message", "part"):
                db.execute(
                    f"UPDATE {table} SET time_created = ?, time_updated = ? WHERE session_id = ?",
                    (stamp, stamp, session),
                )
    sessions = Path(case.env["CODEX_HOME"]) / "sessions"
    for name, session in (("current", current), ("peer", peer)):
        codex_fixture(
            sessions / f"rollout-{session}.jsonl",
            case.work,
            session,
            [("user", bodies[name][2])],
        )
    case.json_command("import", "--provider", "opencode", "--path", fixture.path)
    case.json_command("import", "--provider", "codex", "--path", sessions)

    baseline = _events(case, marker)
    known = {}
    for name, (provider, session, text) in bodies.items():
        matches = [
            event
            for event in baseline.values()
            if event["text"] == text and event["event_type"] == "message"
        ]
        require(len(matches) == 1, f"missing/duplicate full {name} message")
        event = matches[0]
        require(
            event["provider"] == provider and event["provider_session_id"] == session,
            f"wrong {name} identity",
        )
        known[name] = event
    calls = {
        key for key, event in baseline.items() if event["event_type"] == "tool_call"
    }
    outputs = {
        key for key, event in baseline.items() if event["event_type"] == "tool_output"
    }
    require(
        calls and outputs, "fixture must contain both searchable tool calls and outputs"
    )
    for key in calls | outputs:
        event = baseline[key]
        require(
            event["provider"] == "opencode" and event["provider_session_id"] == parent,
            "wrong tool identity",
        )
        if key in calls:
            require(
                event["structured_content"]["state"]["input"] == tool_input,
                "tool call lost complete input",
            )
            require(tool_input["command"] in event["text"], "tool call lost input text")
        else:
            require(event["text"] == tool_output, "tool output lost complete text")
    messages = {event["trwl_event_id"] for event in known.values()}
    all_ids = messages | calls | outputs
    require(set(baseline) == all_ids, "unexpected matching fixture content")
    parent_id = known["parent-a"]["trwl_session_id"]
    child_id = known["child"]["trwl_session_id"]
    require(
        parent_id == known["parent-b"]["trwl_session_id"] and parent_id != child_id,
        "wrong parent/child sessions",
    )
    require(
        case.json_command("locate", "session", child_id)["parent_trwl_session_id"]
        == parent_id,
        "child-parent relationship lost",
    )
    case.check(
        "selection fixture full content, identities, and parent-child relationship"
    )

    def selected(label, expected, *args, all_workspaces=True):
        actual = _events(case, marker, *args, all_workspaces=all_workspaces)
        require(
            set(actual) == expected,
            f"{label}: missing {expected - set(actual)}, unexpected {set(actual) - expected}",
        )
        for key, event in actual.items():
            _same_event(event, baseline[key])
        case.check(label)

    opencode = {
        key for key, event in baseline.items() if event["provider"] == "opencode"
    }
    codex = all_ids - opencode
    child_ids = {known["child"]["trwl_event_id"]}
    parent_ids = (
        {known[name]["trwl_event_id"] for name in ("parent-a", "parent-b")}
        | calls
        | outputs
    )
    selected(
        "provider includes OpenCode and excludes matching Codex records",
        opencode,
        "--provider",
        "opencode",
    )
    selected(
        "provider includes Codex and excludes matching OpenCode records",
        codex,
        "--provider",
        "codex",
    )
    selected(
        "session selects the child and excludes other matching sessions",
        child_ids,
        "--session",
        child_id,
    )
    selected("native session selects the same child", child_ids, "--session", child)
    selected(
        "trwl session prefixes do not select sessions", set(), "--session", child_id[:8]
    )
    selected(
        "native session prefixes do not select sessions", set(), "--session", child[:-1]
    )
    selected(
        "trwl session prefix exclusions do not exclude sessions",
        all_ids,
        "--exclude-session",
        child_id[:8],
    )
    selected(
        "native session prefix exclusions do not exclude sessions",
        all_ids,
        "--exclude-session",
        child[:-1],
    )
    selected(
        "explicit session exclusion keeps the other matching sessions",
        all_ids - child_ids,
        "--exclude-session",
        child_id,
    )
    selected(
        "workspace includes matching local records and excludes the other workspace",
        all_ids - {known["other"]["trwl_event_id"]},
        "--workspace",
        case.work,
        all_workspaces=False,
    )
    selected(
        "date includes recent records and excludes older child/Codex records",
        parent_ids | {known["other"]["trwl_event_id"]},
        "--since",
        "2026-09-22T00:00:00Z",
    )
    selected(
        "all content scope includes every known matching event",
        all_ids,
        "--content-scope",
        "all",
    )
    selected(
        "transcript includes messages and excludes calls/outputs",
        messages,
        "--content-scope",
        "transcript",
    )
    selected(
        "calls includes call input and excludes messages/outputs",
        calls,
        "--content-scope",
        "calls",
    )
    selected(
        "outputs includes full output and excludes messages/calls",
        outputs,
        "--content-scope",
        "outputs",
    )
    selected(
        "default includes child work; primary-only excludes it",
        all_ids - child_ids,
        "--primary-only",
    )

    grouped = _shown(
        case,
        case.search(
            marker, "--workspace", "**", "--provider", "opencode", "--limit", "100"
        )["results"],
    )
    expected_sessions = {
        known[name]["trwl_session_id"] for name in ("parent-a", "child", "other")
    }
    require(
        len(grouped) == len(expected_sessions)
        and {event["trwl_session_id"] for event in grouped.values()}
        == expected_sessions,
        "default grouping did not return one match per exact session",
    )
    for key, event in grouped.items():
        require(key in opencode, "grouping returned an unknown event")
        _same_event(event, baseline[key])
    require(len(grouped) < len(opencode), "grouping did not reduce dense event matches")
    limited = _shown(
        case,
        case.search(
            marker, "--workspace", "**", "--provider", "opencode", "--limit", "2"
        )["results"],
    )
    family = {parent_id, child_id}
    require(
        len(limited) == 2
        and sum(event["trwl_session_id"] in family for event in limited.values()) == 1
        and any(
            event["trwl_session_id"] == known["other"]["trwl_session_id"]
            for event in limited.values()
        ),
        "bounded grouping did not include both independent roots",
    )
    for key, event in limited.items():
        require(key in opencode, "bounded grouping returned an unknown event")
        _same_event(event, baseline[key])
    case.check(
        "session grouping reduces dense matches and diversifies independent roots without fixing winners/order"
    )

    previous = case.env.get("CODEX_THREAD_ID")
    case.env["CODEX_THREAD_ID"] = current
    try:
        selected(
            "detected current Codex session is excluded while its peer remains",
            all_ids - {known["current"]["trwl_event_id"]},
        )
        selected(
            "include-current-session restores the same full content and IDs",
            all_ids,
            "--include-current-session",
        )
    finally:
        if previous is None:
            case.env.pop("CODEX_THREAD_ID")
        else:
            case.env["CODEX_THREAD_ID"] = previous


def updates(case):
    session = "ses_updates"
    codex_session = "550e8400-e29b-41d4-a716-446655440020"
    fixture = OpenCodeFixture(case.providers / "updates.db", case.work)
    fixture.session(session)
    unchanged = (
        "stableorchard Unchanged complete text.\nUnicode: café; punctuation: [] {}."
    )
    original = "oldorchard Original text that must be replaced.\nOriginal second line."
    deleted = "deletedorchard This complete message will be deleted."
    added = "addedorchard Appended assistant reply.\nKeep every line."
    replacement = (
        "revisedorchard Edited replacement.\nThe original body must disappear."
    )
    unrelated = "unrelatedorchard Codex history must survive every OpenCode update.\nUnchanged tail."
    fixture.message(session, "msg_stable", unchanged)
    fixture.message(session, "msg_edit", original)
    fixture.message(session, "msg_delete", deleted)
    codex = codex_fixture(
        case.providers / "unrelated.jsonl",
        case.work,
        codex_session,
        [("user", unrelated)],
    )
    case.json_command("import", "--provider", "codex", "--path", codex)
    case.json_command("import", "--provider", "opencode", "--path", fixture.path)
    stable = _message(case, "stableorchard", unchanged, "opencode", session)
    old = _message(case, "oldorchard", original, "opencode", session)
    removed = _message(case, "deletedorchard", deleted, "opencode", session)
    peer = _message(case, "unrelatedorchard", unrelated, "codex", codex_session)
    require(
        len({event["trwl_event_id"] for event in (stable, old, removed, peer)}) == 4,
        "initial records share event IDs",
    )
    case.check(
        "independent OpenCode SQLite and Codex JSONL histories imported into one data root"
    )

    def retained(*events):
        for event in events:
            _same_event(
                case.json_command("show", "event", event["trwl_event_id"])["event"],
                event,
            )
            found = _events(case, event["text"].split()[0])
            require(
                set(found) == {event["trwl_event_id"]},
                "retained message missing or duplicated in search",
            )
            _same_event(found[event["trwl_event_id"]], event)

    fixture.message(session, "msg_added", added, role="assistant")
    case.json_command("import", "--provider", "opencode", "--path", fixture.path)
    appended = _message(case, "addedorchard", added, "opencode", session)
    require(
        appended["trwl_session_id"] == stable["trwl_session_id"]
        and appended["trwl_event_id"]
        not in {
            stable["trwl_event_id"],
            old["trwl_event_id"],
            removed["trwl_event_id"],
        },
        "append lost identity",
    )
    retained(stable, old, removed, peer)
    case.check(
        "append appears in full; original IDs/content and unrelated history remain"
    )

    with sqlite3.connect(fixture.path) as db:
        stamp = fixture.stamp()
        db.execute(
            "UPDATE part SET data = ?, time_updated = ? WHERE id = ?",
            (json.dumps({"type": "text", "text": replacement}), stamp, "part_msg_edit"),
        )
        db.execute(
            "UPDATE message SET time_updated = ? WHERE id = ?", (stamp, "msg_edit")
        )
        db.execute("UPDATE session SET time_updated = ? WHERE id = ?", (stamp, session))
    case.json_command("import", "--provider", "opencode", "--path", fixture.path)
    edited = _message(case, "revisedorchard", replacement, "opencode", session)
    require(
        edited["trwl_session_id"] == old["trwl_session_id"],
        "edit moved to a different conversation",
    )
    require(not _events(case, "oldorchard"), "replaced text remains searchable")
    listed = case.json_command(
        "list", "events", "--session", stable["trwl_session_id"]
    )["events"]
    require(
        all(original not in event["text"] for event in listed),
        "replaced body remains in the transcript",
    )
    retained(stable, appended, removed, peer)
    case.check(
        "edit replaces complete content; unchanged IDs/text and unrelated history remain"
    )

    with sqlite3.connect(fixture.path) as db:
        db.execute("DELETE FROM part WHERE message_id = ?", ("msg_delete",))
        db.execute("DELETE FROM message WHERE id = ?", ("msg_delete",))
        db.execute(
            "UPDATE session SET time_updated = ? WHERE id = ?",
            (fixture.stamp(), session),
        )
    case.json_command("import", "--provider", "opencode", "--path", fixture.path)
    require(not _events(case, "deletedorchard"), "deleted text remains searchable")
    retained(stable, edited, appended, peer)
    case.check(
        "delete removes searchable content while surviving histories retain IDs/text"
    )

    survivors = (stable, edited, appended)
    expected_ids = {event["trwl_event_id"] for event in survivors}
    for repeat in range(2):
        if repeat:
            case.json_command(
                "import", "--provider", "opencode", "--path", fixture.path
            )
        rows = case.json_command(
            "list", "events", "--session", stable["trwl_session_id"]
        )["events"]
        require(
            len(rows) == len(expected_ids)
            and {event["trwl_event_id"] for event in rows} == expected_ids,
            "update/repeated import left missing or duplicate events",
        )
        for event in survivors:
            _same_event(
                next(
                    row
                    for row in rows
                    if row["trwl_event_id"] == event["trwl_event_id"]
                ),
                event,
            )
        retained(*survivors, peer)
        require(
            not _events(case, "oldorchard") and not _events(case, "deletedorchard"),
            "repeat restored replaced/deleted content",
        )
    case.check(
        "unchanged repeat has no duplicates; exact surviving text/IDs and unrelated history persist"
    )

    def matches_full_import():
        incremental = case.json_command(
            "import", "--provider", "opencode", "--path", fixture.path
        )["sources"][0]
        actual = case.json_command(
            "list", "events", "--workspace", "**", "--limit", "1000"
        )["events"]
        # Keep the incremental index for the next update; full-import checks
        # must not replace the checkpoints whose later resumption they exercise.
        oracle = case.root / "full-import"
        if oracle.exists():
            shutil.rmtree(oracle)
        shutil.copytree(case.data, oracle / "trwl")
        previous = case.env["XDG_DATA_HOME"]
        case.env["XDG_DATA_HOME"] = str(oracle)
        try:
            full = case.json_command(
                "import", "--provider", "opencode", "--path", fixture.path, "--full"
            )["sources"][0]
            expected = case.json_command(
                "list", "events", "--workspace", "**", "--limit", "1000"
            )["events"]
        finally:
            case.env["XDG_DATA_HOME"] = previous
        require(
            actual == expected,
            "incremental import differs from full content, ordering or identities",
        )
        for key in ("events", "rejected_records", "rejection_diagnostics"):
            require(
                incremental[key] == full[key],
                f"incremental {key} differs from full import",
            )

    # A long unchanged beginning mixes one- and two-event parts with a rejected
    # row. Suffix updates must resume event sequences and rejection diagnostics,
    # not infer either from the number of native rows they skip.
    fixture.part(session, "msg_edit", "prefix_bad_part", {})
    with sqlite3.connect(fixture.path) as db:
        db.execute("UPDATE part SET data = '{' WHERE id = 'prefix_bad_part'")
    fixture.part(
        session,
        "msg_edit",
        "prefix_tool",
        {
            "type": "tool",
            "tool": "shell",
            "state": {"input": {"command": "true"}, "output": "prefix result"},
        },
    )
    for n in range(150):
        fixture.message(session, f"msg_prefix_{n}", f"Earlier context {n}.")

    sibling = "ses_updates_sibling"
    fixture.session(sibling)
    fixture.message(sibling, "msg_sibling", "siblingbeacon Initial sibling content.")
    fixture.message(session, "msg_tool", "Inspect the pool.", role="assistant")
    tool = {
        "type": "tool",
        "callID": "call_update",
        "tool": "shell",
        "state": {"status": "running", "input": {"command": "inspect-pool"}},
    }
    fixture.part(session, "msg_tool", "part_tool", tool)
    matches_full_import()
    fixture.message(
        session, "msg_after_tool", "tailbeacon Reply after the unfinished tool."
    )
    for n in range(80):
        fixture.message(session, f"msg_suffix_{n}", f"Later context {n}.")
    matches_full_import()

    # Completion adds an output event before an existing tail. A growing row
    # count must not hide changes to earlier records or their sequence numbers.
    tool["state"].update(
        status="completed", output="poolresultbeacon Active connections: 12."
    )
    with sqlite3.connect(fixture.path) as db:
        db.execute(
            "UPDATE part SET data = ?, time_updated = ? WHERE id = 'part_tool'",
            (json.dumps(tool), fixture.stamp()),
        )
    fixture.message(
        session,
        "msg_completion_tail",
        "completiontailbeacon Continue after tool completion.",
    )
    matches_full_import()
    require(_events(case, "poolresultbeacon"), "completed tool output is missing")

    # One source can append to one session and replace another in the same import.
    with sqlite3.connect(fixture.path) as db:
        db.execute(
            "UPDATE part SET data = ?, time_updated = ? WHERE id = 'part_msg_sibling'",
            (
                json.dumps(
                    {"type": "text", "text": "siblingeditbeacon Edited sibling."}
                ),
                fixture.stamp(),
            ),
        )
    fixture.message(
        session, "msg_mixed_tail", "mixedtailbeacon Append alongside a sibling edit."
    )
    matches_full_import()

    # New part IDs can sort before the saved end, including at an equal timestamp.
    fixture.part(
        session,
        "msg_tool",
        "000_earlier_part",
        {"type": "text", "text": "earlierbeacon Insert before the saved tail."},
    )
    with sqlite3.connect(fixture.path) as db:
        db.execute(
            "UPDATE part SET time_created = (SELECT time_created FROM part WHERE id = 'part_tool') WHERE id = '000_earlier_part'"
        )
    matches_full_import()

    with sqlite3.connect(fixture.path) as db:
        db.execute(
            "UPDATE message SET data = ?, time_updated = ? WHERE id = 'msg_tool'",
            (json.dumps({"role": "user"}), fixture.stamp()),
        )
        db.execute("DELETE FROM part WHERE id = 'part_msg_after_tool'")
    fixture.message(
        session,
        "msg_metadata_tail",
        "metadatatailbeacon Append alongside a role change and deletion.",
    )
    matches_full_import()

    fixture.part(session, "msg_metadata_tail", "bad_part", {})
    with sqlite3.connect(fixture.path) as db:
        db.execute("UPDATE part SET data = '{' WHERE id = 'bad_part'")
    matches_full_import()
    fixture.message(
        session, "msg_after_bad", "rejectiontailbeacon Append after a rejected part."
    )
    matches_full_import()
    with sqlite3.connect(fixture.path) as db:
        db.execute("DELETE FROM part WHERE id = 'bad_part'")
    matches_full_import()

    # Truncating inside the unchanged beginning must remove all later indexed
    # events even though no new native row is available to emit at the old end.
    with sqlite3.connect(fixture.path) as db:
        db.execute(
            "DELETE FROM part WHERE session_id = ? AND time_created >= "
            "(SELECT time_created FROM part WHERE id = 'part_msg_prefix_100')",
            (session,),
        )
    matches_full_import()
    case.check(
        "appends, tool completion, mixed-session edits, insertion order, roles, deletions and diagnostics match full import"
    )


def bad_input(case):
    session = "550e8400-e29b-41d4-a716-446655440030"
    first = (
        "validorchard First valid complete message.\nUnicode: naïve; tail preserved."
    )
    second = (
        "validorchard Second valid complete reply.\nThis follows the malformed row."
    )
    path = codex_fixture(
        case.providers / f"rollout-{session}.jsonl",
        case.work,
        session,
        [("user", first), ("assistant", second)],
    )
    lines = path.read_text().splitlines(keepends=True)
    malformed = '{"timestamp":"2026-09-21T12:00:01Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"badrecordorchard"}\n'
    path.write_text("".join(lines[:2]) + malformed * 25 + "".join(lines[2:]))
    result = case.command(
        "import",
        "--provider",
        "codex",
        "--path",
        path,
        "--format",
        "json",
        "--progress",
        "none",
    )
    report = json.loads(result.stdout)
    diagnostics = [
        item
        for source in report["sources"]
        for item in source.get("rejection_diagnostics", [])
    ]
    require(len(diagnostics) == 10, "import did not cap rejection diagnostics at ten")
    require(
        sum(source.get("rejected_records", 0) for source in report["sources"]) == 25,
        "import did not count all twenty-five rejected records",
    )
    require(
        diagnostics
        and any(
            item.get("path") == str(path)
            and isinstance(item.get("detail"), str)
            and item["detail"].strip()
            for item in diagnostics
        ),
        "mixed input omitted a record without a problem diagnostic",
    )
    require(
        str(path) in result.stdout + result.stderr,
        "mixed-input output does not identify the affected input",
    )
    valid = _events(case, "validorchard")
    require(
        len(valid) == 2
        and {event["text"] for event in valid.values()} == {first, second},
        "malformed row lost or changed a complete valid message",
    )
    for event in valid.values():
        require(
            event["provider"] == "codex"
            and event["provider_session_id"] == session
            and event["event_type"] == "message",
            "wrong retained valid-record identity",
        )
    require(not _events(case, "badrecordorchard"), "malformed message was indexed")
    session_id = next(iter(valid.values()))["trwl_session_id"]
    rows = case.json_command("list", "events", "--session", session_id)["events"]
    require(
        len(rows) == 2 and {event["trwl_event_id"] for event in rows} == set(valid),
        "mixed input transcript contains missing/extra messages",
    )
    case.check(
        "mixed native Codex JSONL identifies a rejected input/problem and keeps both complete valid texts"
    )

    # No valid metadata or event remains in the provider input.
    path.write_text(malformed + "not valid JSON at all\n")
    running = case.start(
        "import",
        "--provider",
        "codex",
        "--path",
        path,
        "--format",
        "json",
        "--progress",
        "none",
    )
    result = case.wait(running, expect_failure=None)
    code = result.returncode
    require(code >= 0, f"invalid-input import terminated by signal: {code}")
    problem = result.stderr
    if code == 0:
        failed = json.loads(result.stdout)
        require(
            failed.get("outcome") == "completed_with_source_failures",
            f"entirely invalid update reported success: {failed.get('outcome')!r}",
        )
        details = [
            source["detail"]
            for source in failed["sources"]
            if source.get("path") == str(path)
            and isinstance(source.get("detail"), str)
            and source["detail"].strip()
        ]
        require(
            details, "unsuccessful update omitted the affected input's problem detail"
        )
        problem += "\n" + "\n".join(details)
    output = result.stdout + result.stderr
    require(str(path) in output, "failed update does not identify the affected input")
    require(
        re.search(
            r"invalid|malformed|pars(?:e|ing)|reject|expected|syntax|schema|\beof\b|no (?:valid|accepted|usable|retained|session)|session_meta|session (?:metadata|header)",
            problem,
            re.IGNORECASE,
        ),
        "failed update does not describe the input problem",
    )
    after = _events(case, "validorchard")
    require(
        set(after) == set(valid), "failed update changed/lost prior valid event IDs"
    )
    for key, event in after.items():
        _same_event(event, valid[key])
    rows = case.json_command("list", "events", "--session", session_id)["events"]
    require(
        len(rows) == 2 and {event["trwl_event_id"] for event in rows} == set(valid),
        "failed update damaged the prior transcript",
    )
    for row in rows:
        _same_event(row, valid[row["trwl_event_id"]])
    require(not _events(case, "badrecordorchard"), "failed input became searchable")
    case.check(
        "entirely invalid update reports failure and prior full texts/session/event IDs remain retrievable"
    )

    # A valid header alone must not turn rejected content into an empty history.
    path.write_text(lines[0] + malformed)
    result = case.command(
        "import",
        "--provider",
        "codex",
        "--path",
        path,
        "--format",
        "json",
        expect_failure=None,
    )
    require(result.returncode >= 0, "malformed replacement terminated by a signal")
    if result.returncode == 0:
        report = json.loads(result.stdout)
        require(
            report.get("outcome") == "completed_with_source_failures",
            "valid header with no accepted content reported success",
        )
    after = _events(case, "validorchard")
    require(
        set(after) == set(valid), "header-only malformed replacement erased history"
    )
    for key, event in after.items():
        _same_event(event, valid[key])
    case.check(
        "valid session metadata with only malformed content preserves prior history"
    )
