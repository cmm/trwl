"""Import and retrieve OpenCode SQLite messages, tools, and a child session."""

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
