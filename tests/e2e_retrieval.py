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
    # The history and update performance runners reuse this verifier. Their
    # source view includes selected parts under the public neutral representation.
    from perf_history import enumerate_history, native_parts

    fixture.part(
        PARENT,
        "msg_answer",
        "part_archive",
        {
            "type": "reasoning",
            "text": "archivebeacon Original selected text.",
            "metadata": {
                "provider": {
                    "signature": "original-signature",
                    "data": [False, 0, None, {}, [], "same", "same"],
                }
            },
        },
    )
    fixture.part(
        PARENT,
        "msg_answer",
        "part_empty_archive",
        {
            "type": "reasoning",
            "text": "",
            "metadata": {"provider": {"signature": "empty-signature"}},
        },
    )
    case.command("import", "--provider", "opencode", "--path", fixture.path)
    parts, _ = native_parts(fixture.path)
    verified = enumerate_history(case, parts, 60)
    require(
        any(
            row["provider_event_id"] == "part_archive"
            and row["text"] == "archivebeacon Original selected text."
            for row in verified
        ),
        "performance verifier missed selected source content",
    )
    texture_retrieval(case)


def texture_retrieval(case):
    session = "texture-layouts"
    path = case.providers / "texture.jsonl"
    ordinary = 'ordinarybeacon thinking reasoning trace. {"type":"thinking"}'
    duplicate = "duplicatebeacon An identical passage."
    late = "plain words " * 100 + "latebeacon exact phrase at the end."
    protected = "`apple` https://example.com ```code```"
    guarded = "a ``code ` apple`` https://guarded.invalid/path?x=ap `unclosed apple"
    markdown_url = "[link](https://example.net/path(with-parens))"
    autolink = "<http://example.net/autolink>"
    url_only = [
        ("uppercase-url", "HTTPS://example.net/uppercase"),
        ("mixed-case-url", "hTtP://example.net/mixed"),
        ("single-quoted-url", "'https://example.net/single-quoted'"),
        ("double-quoted-url", '"HtTpS://example.net/double-quoted"'),
        ("equals-url", "url=HTTPS://example.net/equals"),
        ("unicode-prefix-url", "日本語HTTPS://example.net/unicode-prefix"),
    ]
    protected_urls = [markdown_url, autolink, *(url for _, url in url_only)]
    url_prose = "apple " * 40 + " ".join(protected_urls) + " " + "apple " * 40
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
                {"type": "thinking", "thinking": late},
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
                    "redacted": True,
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
                    "redacted": True,
                    "thinkingSignature": "opaque-image",
                },
                {"type": "image", "data": "original-image"},
            ],
        ),
        ("protected", [{"type": "thinking", "thinking": protected}]),
        ("unicode", [{"type": "thinking", "thinking": "日本語"}]),
        ("blank", [{"type": "thinking", "thinking": ""}]),
        ("tiny", [{"type": "thinking", "thinking": "a"}]),
        ("guarded", [{"type": "thinking", "thinking": guarded}]),
        ("markdown-url", [{"type": "thinking", "thinking": markdown_url}]),
        ("autolink", [{"type": "thinking", "thinking": autolink}]),
        *[(name, [{"type": "thinking", "thinking": url}]) for name, url in url_only],
        ("url-prose", [{"type": "thinking", "thinking": url_prose}]),
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
            "message": {
                "role": "assistant",
                "content": content,
            },
        }
        for name, content in layouts
    ]
    rows[2]["message"].update(thinkingLevel="high", providerThinkingLevel="high")
    rows[2]["message"]["usage"] = {"output": 12, "reasoning": 7}
    write_jsonl(path, [header, *rows[:3]])
    case.command("import", "--provider", "pi", "--path", path)
    write_jsonl(path, [header, *rows])
    selected_path = case.providers / "texture-only.jsonl"
    write_jsonl(selected_path, [{**header, "id": "texture-only"}, rows[0]])
    source_only_path = case.providers / "texture-source-only.jsonl"
    write_jsonl(source_only_path, [{**header, "id": "texture-source-only"}, rows[5]])
    for source in (path, selected_path, source_only_path):
        case.command("import", "--provider", "pi", "--path", source)

    def mode(name):
        if name == "source":
            case.env["TRWL_TEXTURE"] = "source"
        else:
            case.env.pop("TRWL_TEXTURE", None)
        return () if name == "default" else ("--texture",)

    transcripts = {}
    for name in ("default", "expanded", "source"):
        flags = mode(name)
        transcript = case.json_command("show", "session", session, *flags)
        transcripts[name] = transcript
        page_rows, cursor = [], None
        while True:
            page = case.json_command(
                "list",
                "events",
                "--session",
                session,
                "--limit",
                "2",
                *flags,
                *(("--cursor", cursor) if cursor else ()),
            )
            page_rows.extend(page["events"])
            cursor = page["next_cursor"]
            if cursor is None:
                break
        require(
            page_rows == transcript["events"],
            f"{name}: small pages skip or repeat eligible events",
        )
        require(
            len({r["trwl_event_id"] for r in page_rows}) == len(page_rows),
            "duplicate event across pages",
        )
        for position, event in enumerate(page_rows):
            if (
                event["provider_event_id"] == "mixed"
                and event["event_type"] == "message"
            ):
                require(
                    "thinkingLevel" not in event["structured_content"]
                    and "providerThinkingLevel" not in event["structured_content"],
                    "selected mode exposed category-only envelope metadata",
                )
            window = case.json_command(
                "show", "event", event["trwl_event_id"], "--window", "1", *flags
            )
            require(
                window["events"] == page_rows[max(0, position - 1) : position + 2],
                f"{name}: hidden event consumed window radius",
            )
        sessions = case.json_command(
            "list", "sessions", "--provider", "pi", "--limit", "1", *flags
        )
        all_sessions = list(sessions["sessions"])
        while sessions["next_cursor"]:
            sessions = case.json_command(
                "list",
                "sessions",
                "--provider",
                "pi",
                "--limit",
                "1",
                "--cursor",
                sessions["next_cursor"],
                *flags,
            )
            all_sessions.extend(sessions["sessions"])
        only = [r for r in all_sessions if r["provider_session_id"] == "texture-only"]
        require(
            bool(only) == (name != "default"),
            f"{name}: session with no eligible events was listed",
        )
        require(
            any(r["provider_session_id"] == "texture-source-only" for r in all_sessions)
            == (name == "source"),
            "source-only session appeared in another view",
        )
        representative = next(
            r for r in all_sessions if r["provider_session_id"] == session
        )
        require(
            representative["trwl_event_id"] == page_rows[0]["trwl_event_id"],
            "session representative is excluded",
        )
        ordinary_event = next(
            r for r in page_rows if r["provider_event_id"] == "ordinary"
        )
        require(
            ordinary_event["text"] == ordinary
            and ordinary_event["structured_content"] == rows[1]["message"],
            "ordinary category words changed",
        )
        unknown = next(
            row for row in page_rows if row["provider_event_id"] == "unknown"
        )
        require(
            unknown["structured_content"] == rows[-1]["message"]
            and unknown["text"] == "unknownnativebeacon Ordinary unknown part.",
            "unknown Pi forms were reclassified",
        )
        call = next(r for r in page_rows if r["event_type"] == "tool_call")
        require(
            call["structured_content"] == tool,
            "tool input was classified as native detail",
        )
        for native in ("empty", "image"):
            require(
                any(r["provider_event_id"] == native for r in page_rows),
                "empty or non-text ordinary content disappeared",
            )

    def event(name, native):
        return next(
            r
            for r in transcripts[name]["events"]
            if r["provider_event_id"] == native and r["event_type"] == "message"
        )

    default = event("default", "mixed")
    expanded = event("expanded", "mixed")
    source = event("source", "mixed")
    require(
        default["text"] == "ordinarybefore\nordinaryafter\n" + duplicate,
        "default mixed-body projection changed ordinary text",
    )
    require(
        default["structured_content"]["content"]
        == [rows[2]["message"]["content"][i] for i in (0, 2, 4, 5)],
        "default retained native selected nodes",
    )
    native = expanded["structured_content"]["content"]
    require(
        native[3]["type"] == "detail"
        and native[3]["text"] != duplicate
        and native[4]["text"] == duplicate,
        "identical ordinary and selected passages lost provenance",
    )
    require(
        native[1]["text"] != late
        and source["text"]
        == "ordinarybefore\n"
        + late
        + "\nordinaryafter\n"
        + duplicate
        + "\n"
        + duplicate,
        "expanded/source text differs from its mode",
    )
    tiny = event("expanded", "tiny")["text"]
    require(len(tiny) == 1 and tiny != "a", "tiny selected part was not included")
    guarded_text = event("expanded", "guarded")["text"]
    require(
        guarded_text[0] != guarded[0] and guarded_text[1:] == guarded[1:],
        "backtick runs, URLs or unmatched backticks changed",
    )
    for hidden in ("protected", "unicode", "blank", "markdown-url", "autolink"):
        require(
            not any(
                r["provider_event_id"] == hidden
                for r in transcripts["expanded"]["events"]
            ),
            "unchanged-only part consumed an expanded row",
        )
        require(
            event("source", hidden)["structured_content"]["content"][0]["type"]
            == "detail",
            "source mode exposed native discriminator",
        )
    require(
        event("source", "protected")["text"] == protected,
        "protected content was changed in source mode",
    )
    require(
        event("source", "image")["structured_content"]["content"][0]["signature"]
        == "opaque-image",
        "source lost the redacted Pi signature",
    )
    require(
        event("expanded", "first")["structured_content"]["content"][0].keys()
        == {"type", "text"},
        "expanded mode exposed opaque metadata",
    )
    require(
        event("source", "first")["structured_content"]["content"][0]["signature"]
        == "opaque-signature",
        "source signature changed",
    )

    for name in transcripts:
        flags = mode(name)
        if name != "default":
            projected = event(name, "url-prose")["text"]
            changed_urls = [url for url in protected_urls if url not in projected]
            require(
                not changed_urls, f"{name}: protected URLs changed: {changed_urls!r}"
            )
            if name == "expanded":
                require(
                    projected.split(markdown_url)[0] != url_prose.split(markdown_url)[0]
                    and projected.split(autolink)[1] != url_prose.split(autolink)[1],
                    "URL protection swallowed surrounding prose",
                )
            else:
                require(
                    projected == url_prose
                    and event(name, "markdown-url")["text"] == markdown_url
                    and event(name, "autolink")["text"] == autolink,
                    "source URL text changed",
                )
            for native, url in [
                ("markdown-url", markdown_url),
                ("autolink", autolink),
                *url_only,
            ]:
                if name == "expanded":
                    require(
                        not any(
                            row["provider_event_id"] == native
                            for row in transcripts[name]["events"]
                        ),
                        f"expanded: {native} was eligible",
                    )
                else:
                    require(
                        event(name, native)["text"] == url,
                        f"{name}: {native} text changed",
                    )
            hits = case.search("example.net", "--session", session, "--events", *flags)[
                "results"
            ]
            require(
                len(hits) == (1 if name == "expanded" else 3 + len(url_only)),
                "URL-only part has incorrect query eligibility",
            )
            require(
                all(
                    hit["snippet"]
                    in next(
                        row["text"]
                        for row in transcripts[name]["events"]
                        if row["trwl_event_id"] == hit["trwl_event_id"]
                    )
                    for hit in hits
                ),
                "URL snippet differs from its transcript",
            )
        require(
            event(name, "mixed")["structured_content"]["usage"]
            == {"output": 12, "detail_tokens": 7},
            "Pi counter metadata was not neutral",
        )
        hits = case.search("latebeacon", "--events", "--session", session, *flags)[
            "results"
        ]
        require(
            bool(hits) == (name != "default"),
            "selected-only query matched the wrong view",
        )
        if hits:
            require(
                len(hits) == 1
                and hits[0]["snippet"] in event(name, "mixed")["text"]
                and len(hits[0]["snippet"]) <= 400,
                "late snippet recomputed edits after clipping",
            )
        required = case.search(
            "+ordinarybefore +latebeacon", "--events", "--session", session, *flags
        )["results"]
        require(
            bool(required) == (name != "default"), "required terms used different views"
        )
        phrase = case.search(
            '"ordinarybefore ordinaryafter"', "--events", "--session", session, *flags
        )["results"]
        require(
            bool(phrase) == (name == "default"), "phrase did not use the composed view"
        )
        protected_hits = case.search(
            "example.com", "--events", "--session", session, *flags
        )["results"]
        require(
            bool(protected_hits) == (name == "source"),
            "source-only text entered the expanded index",
        )
        for grouped in ((), ("--events",)):
            hits = case.search(
                "alphaonlybeacon ordinarybeacon",
                "--limit",
                "1",
                "--session",
                session,
                *grouped,
                *flags,
            )["results"]
            require(len(hits) == 1, "ineligible winner consumed search limit")
        # Repeated imports cannot change source spans or source identities.
        case.command("import", "--provider", "pi", "--path", path)
        require(
            case.json_command("show", "session", session, *flags) == transcripts[name],
            "repeat import changed projection",
        )

    selected_id = event("expanded", "first")["trwl_event_id"]
    protected_id = event("source", "protected")["trwl_event_id"]
    mode("default")
    for reference in (("event", selected_id), ("session", "texture-only")):
        result = case.command("show", *reference, expect_failure=True)
        require(
            "record reference not found" in result.stderr,
            "excluded reference has a mode-specific diagnostic",
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
    case.command(
        "list",
        "events",
        "--session",
        session,
        "--cursor",
        protected_id,
        "--texture",
        expect_failure=True,
    )
    ordinary_id = event("default", "ordinary")["trwl_event_id"]
    expanded_rows = transcripts["expanded"]["events"]
    position = next(
        i for i, row in enumerate(expanded_rows) if row["trwl_event_id"] == ordinary_id
    )
    continued = case.json_command(
        "list", "events", "--session", session, "--cursor", ordinary_id, "--texture"
    )
    require(
        continued["events"] == expanded_rows[position + 1 :]
        and continued["next_cursor"] is None,
        "mode switch skipped eligible rows after a visible cursor",
    )

    for value in (None, "", "SOURCE", " source", "source"):
        if value is None:
            case.env.pop("TRWL_TEXTURE", None)
        else:
            case.env["TRWL_TEXTURE"] = value
        require(
            not case.search("alphaonlybeacon", "--session", session)["results"],
            "environment alone enabled inclusion",
        )
        flagged = case.search("alphaonlybeacon", "--session", session, "--texture")[
            "results"
        ]
        require(len(flagged) == 1, "flag did not enable original-spelling query")
        shown = case.json_command("--texture", "show", "event", selected_id)
        expected = event("source" if value == "source" else "expanded", "first")
        require(
            shown["event"] == expected and shown["events"] == [expected],
            "flag position or environment value changed projection",
        )

    for name in transcripts:
        flags = mode(name)
        selected = event(name, "mixed")
        for format in ("text", "markdown", "json", "jsonl"):
            args = (
                "show",
                "event",
                selected["trwl_event_id"],
                *flags,
                "--format",
                format,
            )
            rendered = case.command(*args).stdout
            export = case.root / f"texture-{name}.{format}"
            case.command(*args, "--out", export)
            require(
                export.read_text() == rendered, "export differs from stdout projection"
            )
            if format == "json":
                report = json.loads(rendered)
                require(
                    report["event"] == selected and report["events"] == [selected],
                    "JSON bypassed output projection",
                )
            elif format == "jsonl":
                lines = [json.loads(line) for line in rendered.splitlines()]
                require(
                    lines[0]["event"] == selected
                    and lines[-1]["terminal"]
                    and not lines[-1]["truncated"],
                    "JSONL projection or completion changed",
                )
            else:
                require(
                    selected["text"] in rendered,
                    "text or Markdown used a different projection",
                )

    old_data = case.env["XDG_DATA_HOME"]
    case.env["XDG_DATA_HOME"] = str(case.root / "fresh-texture-data")
    case.command("import", "--provider", "pi", "--path", path)
    for name in transcripts:
        flags = mode(name)
        require(
            case.json_command("show", "session", session, *flags) == transcripts[name],
            "fresh import differs from resumed parsing",
        )
    case.env["XDG_DATA_HOME"] = old_data
    mode("source")
    path.rename(path.with_suffix(".indexed-away"))
    require(
        case.json_command("show", "session", session, "--texture")
        == transcripts["source"],
        "source restoration depended on the original file",
    )
    mode("default")
    case.check(
        "native layouts, provenance, clipping, eligibility, modes, formats and pagination share one projection"
    )
