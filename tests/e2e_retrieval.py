"""Import and retrieve OpenCode SQLite messages, tools, and a child session."""

import json

from e2e_providers import write_jsonl
from e2e_support import OpenCodeFixture, require

PARENT = "ses_fixture_parent"
CHILD = "ses_fixture_child"
BEFORE = "beforebeacon Check the active database connections before changing the pool."
TEXT = "quartzorchard The database connection pool exhausts under concurrent requests."
AFTER = (
    "parentanswer Inspect the idle timeout and release every checked-out connection."
)
CHILD_TEXT = "childbeacon Increase the connection pool and release idle connections."
TOOL_INPUT = {"command": "printf callbeacon"}
TOOL_OUTPUT = "outputbeacon Pool inspection finished.\nActive connections: 12; idle connections: 3."


def retrieval(case):
    fixture = OpenCodeFixture(case.providers / "opencode.db", case.work)
    fixture.session(PARENT)
    fixture.session(CHILD, parent=PARENT, agent="explore")
    fixture.message(PARENT, "msg_before", BEFORE)
    fixture.message(PARENT, "msg_parent", TEXT)
    fixture.message(PARENT, "msg_answer", AFTER, role="assistant")
    fixture.message(CHILD, "msg_child", CHILD_TEXT, role="assistant")
    # Exercise output attached to a tool call and a separate tool-result part.
    fixture.part(
        PARENT,
        "msg_answer",
        "part_tool",
        {
            "type": "tool",
            "callID": "call_fixture",
            "tool": "bash",
            "state": {
                "status": "completed",
                "input": TOOL_INPUT,
                "output": TOOL_OUTPUT,
            },
        },
    )
    fixture.part(
        PARENT,
        "msg_answer",
        "part_result",
        {
            "type": "tool_result",
            "callID": "call_fixture",
            "text": TOOL_OUTPUT,
        },
    )
    case.json_command("import", "--provider", "opencode", "--path", fixture.path)
    case.check("import OpenCode SQLite fixture")

    def message(marker, expected_text, provider_session):
        hits = case.search(marker, "--events")["results"]
        require(
            len(hits) == 1, f"expected one message hit for {marker}, got {len(hits)}"
        )
        hit = hits[0]
        event = case.json_command("show", "event", hit["trwl_event_id"])["event"]
        require(
            event["text"] == expected_text,
            f"show changed complete {marker} message: {event['text']!r}",
        )
        require(
            event["trwl_event_id"] == hit["trwl_event_id"]
            and event["trwl_session_id"] == hit["trwl_session_id"],
            "search/show identity differs",
        )
        require(
            event["provider"] == "opencode"
            and event["provider_session_id"] == provider_session
            and event["event_type"] == "message",
            f"wrong message identity for {marker}",
        )
        location = case.json_command("locate", "event", hit["trwl_event_id"])
        require(
            location["trwl_event_id"] == hit["trwl_event_id"]
            and location["trwl_session_id"] == hit["trwl_session_id"]
            and location["provider"] == "opencode"
            and location["provider_session_id"] == provider_session,
            "locate event identity differs",
        )
        return event, location

    before, _ = message("beforebeacon", BEFORE, PARENT)
    selected, location = message("quartzorchard", TEXT, PARENT)
    after, _ = message("parentanswer", AFTER, PARENT)
    child, child_location = message("childbeacon", CHILD_TEXT, CHILD)
    parent_id = selected["trwl_session_id"]
    child_id = child["trwl_session_id"]
    require(
        before["trwl_session_id"] == after["trwl_session_id"] == parent_id,
        "parent messages split across sessions",
    )
    require(child_id != parent_id, "child and parent collapsed")
    child_session = case.json_command("locate", "session", child_id)
    require(
        child_session["provider"] == "opencode"
        and child_session["provider_session_id"] == CHILD
        and child_session["parent_trwl_session_id"] == parent_id,
        "child-parent relationship lost",
    )
    case.check("complete messages and child-parent identity via search/show/locate")

    window = case.json_command(
        "show", "event", selected["trwl_event_id"], "--window", "1"
    )
    for event in (before, selected, after):
        require(
            any(
                row["trwl_event_id"] == event["trwl_event_id"]
                and row["text"] == event["text"]
                for row in window["events"]
            ),
            f"window misses expected message {event['text']!r}",
        )
    case.check("window contains selected message and both neighbors")

    for marker, scope, kind in [
        ("callbeacon", "calls", "tool_call"),
        ("outputbeacon", "outputs", "tool_output"),
    ]:
        hits = case.search(marker, "--events", "--content-scope", scope)["results"]
        require(hits, f"missing {kind} search results")
        matching = []
        for hit in hits:
            event = case.json_command("show", "event", hit["trwl_event_id"])["event"]
            require(
                event["trwl_event_id"] == hit["trwl_event_id"]
                and event["trwl_session_id"] == hit["trwl_session_id"],
                "tool search/show identity differs",
            )
            if event["event_type"] == kind:
                matching.append(event)
        require(matching, f"search returned no {kind} event")
        for event in matching:
            require(
                event["provider"] == "opencode"
                and event["provider_session_id"] == PARENT
                and event["trwl_session_id"] == parent_id,
                f"wrong {kind} session",
            )
            if kind == "tool_call":
                require(
                    event["structured_content"]["state"]["input"] == TOOL_INPUT,
                    "show lost complete tool input object",
                )
                require(
                    event["structured_content"]["state"]["output"] == TOOL_OUTPUT,
                    "show lost output attached to completed tool call",
                )
                require(
                    TOOL_INPUT["command"] in event["text"],
                    "show lost complete tool input text",
                )
            else:
                require(event["text"] == TOOL_OUTPUT, "show lost complete tool output")
    case.check("complete tool input/output through show, distinguished by event type")

    listed = case.json_command("list", "events", "--session", parent_id)["events"]
    transcript = case.json_command("show", "session", parent_id)
    # Tool call/output records share timestamps. Window ordering must agree with
    # the complete transcript at ties and at both ends of the session.
    for position, event in enumerate(transcript["events"]):
        neighbors = case.json_command(
            "show", "event", event["trwl_event_id"], "--window", "1"
        )
        require(
            neighbors["events"]
            == transcript["events"][max(0, position - 1) : position + 2],
            "window differs from transcript at a tie or session boundary",
        )
    for reference in (PARENT, parent_id[:12]):
        require(
            case.json_command("show", "session", reference) == transcript,
            "session reference changed transcript or representative location",
        )
    require(
        case.json_command("locate", "session", parent_id)
        == case.json_command(
            "locate", "event", transcript["events"][0]["trwl_event_id"]
        ),
        "session location does not identify its first event",
    )
    for event in (before, selected, after):
        for rows in (listed, transcript["events"]):
            require(
                any(
                    row["trwl_event_id"] == event["trwl_event_id"]
                    and row["text"] == event["text"]
                    for row in rows
                ),
                "list/session lost complete parent message",
            )
    text = case.command("show", "session", parent_id, "--format", "text").stdout
    export = case.root / "transcript.md"
    case.command(
        "show",
        "session",
        parent_id,
        "--format",
        "markdown",
        "--out",
        export,
    )
    markdown = export.read_text()
    # Accept display indentation; JSON retrieval above checks the exact text.
    text = "\n".join(line.lstrip() for line in text.splitlines())
    for body in (BEFORE, TEXT, AFTER, TOOL_INPUT["command"], TOOL_OUTPUT):
        require(body in text, f"text transcript loses {body!r}")
        require(body in markdown, f"Markdown export loses {body!r}")
    case.check("list, session, text transcript and Markdown export preserve content")

    for value in (location, child_location, child_session):
        require(
            value["source"]["source_format"] == "opencode_sqlite",
            "wrong locate source format",
        )
    for value, native_id in (
        (location, "part_msg_parent"),
        (child_location, "part_msg_child"),
    ):
        require(
            value["provider_event_id"] == native_id,
            f"locate identifies the wrong provider record: {value!r}",
        )
    case.check("locate identifies the original provider session and record")
    # The history and update performance runners reuse this verifier.
    from perf_history import enumerate_history, native_parts

    for part, text in (
        ("part_archive", "archivebeacon Selected text."),
        ("part_empty_archive", ""),
    ):
        fixture.part(
            PARENT,
            "msg_answer",
            part,
            {
                "type": "reasoning",
                "text": text,
                "metadata": {
                    "provider": {
                        "signature": "selected-signature",
                        "data": [False, 0, None, {}, []],
                    }
                },
            },
        )
    case.command("import", "--provider", "opencode", "--path", fixture.path)
    parts, _ = native_parts(fixture.path)
    verified = enumerate_history(case, parts, 60)
    require(
        not any(row["provider_event_id"] == "part_archive" for row in verified),
        "history verifier included selected content",
    )
    redacted_retrieval(case)


def redacted_retrieval(case):
    session = "redacted-layouts"
    path = case.providers / "redacted.jsonl"
    ordinary = 'ordinarybeacon thinking reasoning trace. {"type":"thinking"}'
    duplicate = "duplicatebeacon An identical passage."
    tool = {
        "type": "toolCall",
        "id": "call",
        "name": "read",
        "arguments": {
            "type": "thinking",
            "reasoning": "toolbeacon Original tool data.",
            "thinkingLevel": "tool-owned",
            "providerThinkingLevel": "tool-owned",
        },
    }
    layouts = [
        (
            "first",
            [
                {
                    "type": "thinking",
                    "thinking": "alphaonlybeacon Selected content.",
                    "thinkingSignature": "opaque-signature",
                }
            ],
        ),
        ("ordinary", [{"type": "text", "text": ordinary}]),
        (
            "mixed",
            [
                {"type": "text", "text": "ordinarybefore"},
                {
                    "type": "thinking",
                    "thinking": "plain words " * 100 + "latebeacon exact phrase.",
                },
                {"type": "text", "text": "ordinaryafter"},
                {"type": "thinking", "thinking": duplicate},
                {"type": "text", "text": duplicate},
                tool,
            ],
        ),
        (
            "empty",
            [
                {
                    "type": "thinking",
                    "thinking": "",
                    "thinkingSignature": "opaque-empty",
                },
                {"type": "text", "text": ""},
            ],
        ),
        (
            "image",
            [
                {
                    "type": "thinking",
                    "thinking": "",
                    "thinkingSignature": "opaque-image",
                },
                {"type": "image", "data": "original-image"},
            ],
        ),
        (
            "protected",
            [
                {
                    "type": "thinking",
                    "thinking": "`apple` https://example.com ```code```",
                }
            ],
        ),
        ("unicode", [{"type": "thinking", "thinking": "日本語"}]),
        ("blank", [{"type": "thinking", "thinking": ""}]),
        # These are negative adapter inputs, not claims about Pi persistence.
        (
            "unknown",
            [
                {
                    "type": "reasoning",
                    "text": "unknownnativebeacon Ordinary unknown part.",
                },
                {"type": "redacted_thinking", "data": "unknown opaque value"},
            ],
        ),
    ]
    header = {"type": "session", "version": 3, "id": session, "cwd": str(case.work)}
    rows = [
        {
            "type": "message",
            "id": name,
            "timestamp": "2026-10-01T00:00:00Z",
            "message": {"role": "assistant", "content": content},
        }
        for name, content in layouts
    ]
    rows[2]["message"].update(
        thinkingLevel="high",
        providerThinkingLevel="high",
        usage={"output": 12, "reasoning": 7},
    )
    write_jsonl(path, [header, *rows[:3]])
    case.command("import", "--provider", "pi", "--path", path)
    write_jsonl(path, [header, *rows])
    case.command("import", "--provider", "pi", "--path", path)
    selected_path = case.providers / "redacted-only.jsonl"
    write_jsonl(selected_path, [{**header, "id": "redacted-only"}, rows[0]])
    case.command("import", "--provider", "pi", "--path", selected_path)

    transcript = case.json_command("show", "session", session)
    page_rows, cursor = [], None
    while True:
        page = case.json_command(
            "list",
            "events",
            "--session",
            session,
            "--limit",
            "2",
            *(("--cursor", cursor) if cursor else ()),
        )
        page_rows.extend(page["events"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    require(
        page_rows == transcript["events"], "small pages skip or repeat ordinary events"
    )
    require(
        len({r["trwl_event_id"] for r in page_rows}) == len(page_rows),
        "duplicate event across pages",
    )
    for position, event in enumerate(page_rows):
        window = case.json_command(
            "show", "event", event["trwl_event_id"], "--window", "1"
        )
        require(
            window["events"] == page_rows[max(0, position - 1) : position + 2],
            "redacted event consumed window radius",
        )

    def event(native):
        return next(
            r
            for r in page_rows
            if r["provider_event_id"] == native and r["event_type"] == "message"
        )

    require(
        event("ordinary")["text"] == ordinary
        and event("ordinary")["structured_content"] == rows[1]["message"],
        "ordinary category words changed",
    )
    require(
        event("unknown")["structured_content"] == rows[-1]["message"],
        "unknown Pi forms were reclassified",
    )
    require(
        next(r for r in page_rows if r["event_type"] == "tool_call")[
            "structured_content"
        ]
        == tool,
        "tool input was classified as thinking",
    )
    require(
        event("empty")["text"] == ""
        and event("image")["structured_content"]["content"]
        == [{"type": "image", "data": "original-image"}],
        "empty or non-text ordinary content disappeared",
    )
    mixed = event("mixed")
    require(
        mixed["text"] == "ordinarybefore\nordinaryafter\n" + duplicate,
        "redaction changed ordinary text",
    )
    require(
        mixed["structured_content"]["content"]
        == [rows[2]["message"]["content"][i] for i in (0, 2, 4, 5)],
        "redaction changed native siblings",
    )
    require(
        mixed["structured_content"]["usage"] == {"output": 12, "detail_tokens": 7},
        "Pi counter metadata changed",
    )
    require(
        "thinkingLevel" not in mixed["structured_content"]
        and "providerThinkingLevel" not in mixed["structured_content"],
        "category-only metadata exposed",
    )
    for marker in (
        "alphaonlybeacon",
        "latebeacon",
        "example.com",
        "+ordinarybefore +latebeacon",
    ):
        require(
            not case.search(marker, "--events", "--session", session)["results"],
            f"selected text was searchable: {marker}",
        )
    require(
        case.search('"ordinarybefore ordinaryafter"', "--events", "--session", session)[
            "results"
        ],
        "phrase did not use ordinary text",
    )
    for grouped in ((), ("--events",)):
        hits = case.search(
            "alphaonlybeacon ordinarybeacon",
            "--limit",
            "1",
            "--session",
            session,
            *grouped,
        )["results"]
        require(
            len(hits) == 1
            and hits[0]["trwl_event_id"] == event("ordinary")["trwl_event_id"],
            "redacted winner consumed search limit",
        )
    sessions = case.json_command("list", "sessions", "--provider", "pi")["sessions"]
    require(
        not any(r["provider_session_id"] == "redacted-only" for r in sessions),
        "thinking-only session was listed",
    )

    # Identity remains metadata-only even though the event cannot be displayed.
    location = case.json_command("locate", "session", session)
    selected_id = location["trwl_event_id"]
    require(
        case.json_command("locate", "event", selected_id) == location,
        "metadata references disagree",
    )
    require(
        location["provider_event_id"] == "first" and "text" not in location,
        "redaction lost event identity",
    )
    for reference in (
        ("event", selected_id),
        ("event", selected_id, "--window", "1"),
        ("session", "redacted-only"),
    ):
        result = case.command("show", *reference, expect_failure=True)
        require(
            "record reference not found" in result.stderr,
            "redacted reference has a different diagnostic",
        )
    case.command(
        "list",
        "events",
        "--session",
        session,
        "--cursor",
        selected_id,
        expect_failure=True,
    )
    for value in ("", "SOURCE", "source"):
        case.env["TRWL_TEXTURE"] = value
        require(
            case.json_command("show", "session", session) == transcript,
            "removed environment setting changed retrieval",
        )
        case.command("show", "session", session, "--texture", expect_failure=True)
    case.env.pop("TRWL_TEXTURE", None)

    for format in ("text", "markdown", "json", "jsonl"):
        args = ("show", "event", mixed["trwl_event_id"], "--format", format)
        rendered = case.command(*args).stdout
        export = case.root / f"redacted.{format}"
        case.command(*args, "--out", export)
        require(export.read_text() == rendered, "export differs from stdout")
        require(
            "latebeacon" not in rendered and "opaque-signature" not in rendered,
            "export exposed selected content",
        )
        if format == "json":
            report = json.loads(rendered)
            require(
                report["event"] == mixed and report["events"] == [mixed],
                "JSON bypassed redaction",
            )
        elif format == "jsonl":
            lines = [json.loads(line) for line in rendered.splitlines()]
            require(
                lines[0]["event"] == mixed
                and lines[-1]["terminal"]
                and not lines[-1]["truncated"],
                "JSONL redaction or completion changed",
            )
        else:
            require(mixed["text"] in rendered, "human output used a different body")
    case.command("import", "--provider", "pi", "--path", path)
    require(
        case.json_command("show", "session", session) == transcript,
        "repeat import changed retrieval",
    )
    old_data = case.env["XDG_DATA_HOME"]
    case.env["XDG_DATA_HOME"] = str(case.root / "fresh-redacted-data")
    case.command("import", "--provider", "pi", "--path", path)
    require(
        case.json_command("show", "session", session) == transcript,
        "fresh import differs from resumed parsing",
    )
    case.env["XDG_DATA_HOME"] = old_data
    path.rename(path.with_suffix(".indexed-away"))
    require(
        case.json_command("show", "session", session) == transcript,
        "redaction depended on the source file",
    )
    case.check(
        "thinking omitted across retrieval, formats, pagination and incremental imports; ordinary siblings preserved"
    )
