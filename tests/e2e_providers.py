"""Native coding-harness histories, checked through the public CLI."""

import json
from copy import deepcopy

from e2e_support import OpenCodeFixture, require


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )


def native_records(provider, workspace, session):
    """Pi session-v3 and Claude Code project transcripts, including tool activity."""
    prompt = f"{provider}promptbeacon Inspect the parser.\nPreserve 日本語 and résumé."
    command = f"echo {provider}inputbeacon"
    output = f"{provider}outputbeacon First output line.\nComplete second line: λ."
    reply = f"{provider}replybeacon The parser now handles empty input."
    contents = [
        ("user", [{"type": "text", "text": prompt}]),
        (
            "assistant",
            [
                {
                    "type": "toolCall" if provider == "pi" else "tool_use",
                    "id": "call_native",
                    "name": "bash" if provider == "pi" else "Bash",
                    "arguments" if provider == "pi" else "input": {"command": command},
                }
            ],
        ),
        (
            "toolResult" if provider == "pi" else "user",
            [{"type": "text", "text": output}]
            if provider == "pi"
            else [
                {"type": "tool_result", "tool_use_id": "call_native", "content": output}
            ],
        ),
        ("assistant", [{"type": "text", "text": reply}]),
    ]
    records = []
    if provider == "pi":
        records.append(
            {
                "type": "session",
                "version": 3,
                "id": session,
                "timestamp": "2026-09-21T12:00:00.000Z",
                "cwd": str(workspace),
            }
        )
    for index, (role, content) in enumerate(contents, 1):
        message = {"role": role, "content": content}
        if role == "toolResult":
            message.update(toolCallId="call_native", toolName="bash", isError=False)
        common = {
            "timestamp": f"2026-09-21T12:00:0{index}.000Z",
            "message": message,
        }
        parent = f"message-{index - 1}" if index > 1 else None
        records.append(
            {**common, "type": "message", "id": f"message-{index}", "parentId": parent}
            if provider == "pi"
            else {
                **common,
                "type": role,
                "uuid": f"message-{index}",
                "parentUuid": parent,
                "sessionId": session,
                "cwd": str(workspace),
                "version": "test",
                "isSidechain": False,
            }
        )
    expected = [
        (f"{provider}promptbeacon", prompt, "message", "user"),
        (f"{provider}inputbeacon", command, "tool_call", None),
        (f"{provider}outputbeacon", output, "tool_output", None),
        (f"{provider}replybeacon", reply, "message", "assistant"),
    ]
    return records, expected


def check_history(case, provider, session, expected, previous=None):
    identities = {}
    selector = "copilot-cli" if provider == "copilot_cli" else provider
    for index, (marker, body, kind, role) in enumerate(expected, 1):
        hits = case.search(marker, "--provider", selector, "--events")["results"]
        require(len(hits) == 1, f"{provider}: missing or duplicate {marker}")
        hit = hits[0]
        event = case.json_command("show", "event", hit["trwl_event_id"])["event"]
        require(event["event_type"] == kind, f"{marker}: wrong event type")
        if role is not None:
            require(event["role"] == role, f"{marker}: wrong message role")
        # Check native content, not generated labels or JSON object key order.
        require(
            event["text"] == body
            if kind == "message"
            else body in event["text"]
            or json.dumps(body, ensure_ascii=False)
            in json.dumps(event.get("structured_content"), ensure_ascii=False),
            f"{marker}: complete content changed",
        )
        require(
            event["occurred_at"] == hit["timestamp"]
            and event["occurred_at"] == f"2026-09-21T12:00:0{index}.000Z",
            f"{marker}: native timestamp lost",
        )
        location = case.json_command("locate", "event", hit["trwl_event_id"])
        identity = {
            "trwl_event_id": hit["trwl_event_id"],
            "trwl_session_id": hit["trwl_session_id"],
            "provider": provider,
            "provider_session_id": session,
        }
        for value in (event, location):
            require(
                all(
                    value.get(key) == expected_value
                    for key, expected_value in identity.items()
                ),
                f"{marker}: search/show/locate disagree about the record",
            )
        identities[marker] = (hit["trwl_event_id"], hit["trwl_session_id"])
        if previous and marker in previous:
            require(
                identities[marker] == previous[marker], f"{marker}: reference changed"
            )
    sessions = {session_id for _, session_id in identities.values()}
    require(len(sessions) == 1, "native conversation split across sessions")
    ids = [event_id for event_id, _ in identities.values()]
    require(len(set(ids)) == len(ids), "distinct native events collapsed")
    transcript = case.json_command("show", "session", sessions.pop())
    # Select our known content events, not metadata notices or incidental rows.
    require(
        [
            row["trwl_event_id"]
            for row in transcript["events"]
            if row["trwl_event_id"] in ids
        ]
        == ids,
        "transcript lost conversation members or their chronology",
    )
    return identities


def native_jsonl(case, provider):
    session = f"{provider}-native-session"
    root = case.providers / provider / "sessions"
    path = root / "-workspace" / f"{session}.jsonl"
    records, expected = native_records(provider, case.work, session)
    write_jsonl(path, records)
    args = (
        "import",
        "--provider",
        provider,
        "--path",
        path if provider == "pi" else root,
    )
    case.command(*args)
    identities = check_history(case, provider, session, expected)
    case.command(*args)
    check_history(case, provider, session, expected, identities)
    case.check(
        "messages and tool activity: full content, roles, times, conversation and source identity"
    )
    case.check("unchanged import preserves references without duplicates")

    added = f"{provider}appendbeacon A later user message."
    row = dict(records[1] if provider == "pi" else records[0])
    row.update(
        timestamp="2026-09-21T12:00:05.000Z", message={"role": "user", "content": added}
    )
    if provider == "pi":
        row.update(id="message-5", parentId="message-4")
    else:
        row.update(uuid="message-5", parentUuid="message-4")
    write_jsonl(path, [*records, row])
    appended = [*expected, (f"{provider}appendbeacon", added, "message", "user")]
    case.command(*args)
    identities = check_history(case, provider, session, appended, identities)
    case.check("append imports new content and preserves existing references")

    corrected = f"{provider}correctedbeacon The corrected parser explanation."
    records[-1]["message"]["content"] = [{"type": "text", "text": corrected}]
    write_jsonl(path, records)
    revised = [
        *expected[:-1],
        (f"{provider}correctedbeacon", corrected, "message", "assistant"),
    ]
    case.command(*args)
    check_history(case, provider, session, revised, identities)
    for marker in (f"{provider}replybeacon", f"{provider}appendbeacon"):
        require(
            not case.search(marker, "--events")["results"],
            f"{marker}: stale content remains",
        )
    case.check(
        "rewrite replaces corrected and deleted text without changing surviving references"
    )


def pi_jsonl(case):
    native_jsonl(case, "pi")


def claude_code_jsonl(case):
    native_jsonl(case, "claude")


def copilot_records(workspace, session):
    expected = [
        (
            "copilotpromptbeacon",
            "copilotpromptbeacon Inspect the parser.\nKeep 日本語.",
            "message",
            "user",
        ),
        (
            "copilotreplybeacon",
            "copilotreplybeacon The parser handles empty input.",
            "message",
            "assistant",
        ),
    ]
    records = [
        {
            "id": "session-start",
            "type": "session.start",
            "timestamp": "2026-09-21T12:00:00.000Z",
            "data": {
                "sessionId": session,
                "startTime": "2026-09-21T12:00:00.000Z",
                "context": {"cwd": str(workspace)},
            },
        }
    ]
    for index, (_, body, _, role) in enumerate(expected, 1):
        records.append(
            {
                "id": f"message-{index}",
                "type": f"{role}.message",
                "timestamp": f"2026-09-21T12:00:0{index}.000Z",
                "data": {"content": body},
            }
        )
    return records, expected


def copilot_cli_jsonl(case):
    session = "copilot-native-session"
    root = case.providers / "copilot/session-state"
    path = root / session / "events.jsonl"
    records, expected = copilot_records(case.work, session)
    write_jsonl(path, records)
    args = ("import", "--provider", "copilot-cli", "--path", root)
    identities = None
    for _ in range(2):
        case.command(*args)
        identities = check_history(case, "copilot_cli", session, expected, identities)
    case.check(
        "Copilot CLI messages retain complete content, times and references across repeated imports"
    )
    added = "copilotappendbeacon A later message."
    row = {
        "id": "message-3",
        "type": "user.message",
        "timestamp": "2026-09-21T12:00:03.000Z",
        "data": {"content": added},
    }
    write_jsonl(path, [*records, row])
    case.command(*args)
    identities = check_history(
        case,
        "copilot_cli",
        session,
        [*expected, ("copilotappendbeacon", added, "message", "user")],
        identities,
    )
    corrected = "copilotcorrectedbeacon The corrected explanation."
    records[-1]["data"]["content"] = corrected
    write_jsonl(path, records)
    case.command(*args)
    check_history(
        case,
        "copilot_cli",
        session,
        [expected[0], ("copilotcorrectedbeacon", corrected, "message", "assistant")],
        identities,
    )
    for marker in ("copilotreplybeacon", "copilotappendbeacon"):
        require(
            not case.search(marker, "--events")["results"],
            f"{marker}: stale content remains",
        )
    case.check(
        "Copilot CLI append, correction and deletion preserve surviving references"
    )
    copilot_details(case)


def copilot_details(case):
    # Copilot SDK AssistantMessageData, AssistantReasoningEvent and
    # AssistantMessageReasoningBlocks, including their persisted event envelope.
    session = "copilot-details"
    path = case.providers / "copilot-details/events.jsonl"
    identity = lambda n: f"00000000-0000-4000-8000-{n:012d}"

    def native(n, kind, data):
        return {
            "id": identity(n),
            "parentId": identity(n - 1) if n else None,
            "timestamp": "2026-10-01T00:00:00.000Z",
            "type": kind,
            "data": data,
        }

    duplicate = "duplicatecopilotbeacon Identical ordinary and selected text."
    ordinary = duplicate + '\nthinking reasoning trace {"type":"thinking"}'
    sidecar = "plain words " * 100 + "sidecarlatebeacon Selected sidecar text."
    replay_text = {"type": "text", "text": "Original replay text with reasoning words."}
    tool = {
        "type": "tool_use",
        "id": "tool",
        "name": "read",
        "input": {
            "type": "thinking",
            "reasoningText": "tooljsonbeacon Original arguments.",
        },
    }
    blocks = [
        {"type": "thinking", "thinking": duplicate, "signature": "block-signature"},
        {"type": "redacted_thinking", "data": "redacted-opaque"},
        {
            "type": "reasoning",
            "summary": [
                {
                    "type": "summary_text",
                    "text": "summarycopilotbeacon Original summary.",
                }
            ],
            "content": [
                {
                    "type": "reasoning_text",
                    "text": "contentcopilotbeacon Original native content.",
                }
            ],
            "encrypted_content": "native-ciphertext",
        },
        replay_text,
        *[
            {
                "type": "thinking",
                "thinking": f"arraycopilotbeacon{n} Selected array entry.",
            }
            for n in range(12)
        ],
    ]
    ordered = [
        replay_text,
        {"type": "thinking", "thinking": duplicate, "signature": "ordered-signature"},
        tool,
        {
            "type": "thinking",
            "thinking": "`protectedcopilotbeacon`",
            "signature": "protected-signature",
        },
        {
            "type": "unknown_native_form",
            "payload": {"type": "thinking", "thinking": "unknown payload"},
        },
    ]
    mixed = {
        "messageId": identity(102),
        "content": ordinary,
        "reasoningText": sidecar,
        "reasoningOpaque": "sidecar-signature",
        "encryptedContent": "sidecar-ciphertext",
        "reasoningWireField": "reasoning_content",
        "phase": "thinking",
        "reasoningBlocks": {
            "provider": "anthropic",
            "blocks": blocks,
            "orderedBlocks": ordered,
            "revision": 7,
        },
        "details": {"reasoningText": "Original extension field."},
        "toolRequests": [
            {"toolCallId": "tool", "name": "read", "arguments": tool["input"]}
        ],
        "serverTools": {"provider": "anthropic", "rawContentBlocks": [tool]},
    }
    records = [
        native(
            0,
            "session.start",
            {
                "sessionId": session,
                "startTime": "2026-10-01T00:00:00Z",
                "context": {"cwd": str(case.work)},
            },
        ),
        native(
            1,
            "user.message",
            {
                "messageId": identity(101),
                "content": "copilotordinarybeacon Ordinary prompt.",
            },
        ),
        native(2, "assistant.message", mixed),
        native(
            3,
            "assistant.reasoning",
            {
                "reasoningId": identity(203),
                "content": "standalonecopilotbeacon Selected event.",
            },
        ),
        native(4, "assistant.reasoning", {"reasoningId": identity(204), "content": ""}),
        native(
            5,
            "assistant.message",
            {
                "messageId": identity(105),
                "content": "",
                "reasoningOpaque": "opaqueonly-signature",
                "encryptedContent": "opaqueonly-ciphertext",
                "phase": "response",
            },
        ),
        native(
            6,
            "assistant.message",
            {
                "messageId": identity(106),
                "content": "phasestillordinarybeacon Original scalar content.",
                "phase": "thinking",
            },
        ),
        native(
            7,
            "assistant.reasoning_delta",
            {"reasoningId": identity(207), "deltaContent": "transientcopilotbeacon"},
        ),
    ]
    write_jsonl(path, records[:4])
    case.command("import", "--provider", "copilot-cli", "--path", path)
    write_jsonl(path, records)
    case.command("import", "--provider", "copilot-cli", "--path", path)

    def mode(name):
        case.env.pop("TRWL_TEXTURE", None)
        if name == "source":
            case.env["TRWL_TEXTURE"] = "source"
        return () if name == "default" else ("--texture",)

    def transcript(name):
        return case.json_command("show", "session", session, *mode(name))["events"]

    views = {name: transcript(name) for name in ("default", "expanded", "source")}

    def event(name, n):
        return next(
            row for row in views[name] if row["provider_event_id"] == identity(n)
        )

    expected_native = {
        key: value
        for key, value in mixed.items()
        if key
        not in (
            "reasoningText",
            "reasoningOpaque",
            "encryptedContent",
            "reasoningWireField",
            "reasoningBlocks",
        )
    }
    expected_native["phase"] = "detail"
    for name, rows in views.items():
        shown = event(name, 2)
        require(
            shown["structured_content"] == expected_native,
            "Copilot projection changed ordinary fields or retained selected sidecars",
        )
        archive = shown["details"][-1]
        require(
            archive["type"] == "archive"
            and archive["provider"] == "anthropic"
            and archive["revision"] == 7,
            "native block envelope was not relocated neutrally",
        )
        if name == "default":
            require(
                shown["text"] == ordinary
                and shown["details"]
                == [
                    {
                        "type": "archive",
                        "provider": "anthropic",
                        "revision": 7,
                        "blocks": [replay_text],
                        "orderedBlocks": [ordered[i] for i in (0, 2, 4)],
                    }
                ],
                "default replay projection changed ordinary entries",
            )
        else:
            require(
                shown["details"][0]["type"] == "detail"
                and shown["details"][0]["text"] != "",
                "sidecar detail was not relocated",
            )
            require(
                archive["blocks"][3 if name == "source" else 2] == replay_text
                and archive["orderedBlocks"][0] == replay_text
                and archive["orderedBlocks"][2] == tool,
                "mixed replay text/tool entry changed",
            )
        require(
            not any(row["provider_event_id"] == identity(7) for row in rows),
            "transient delta was imported",
        )
        require(
            any(row["provider_event_id"] == identity(4) for row in rows)
            == (name == "source"),
            "empty standalone event has incorrect availability",
        )
        require(
            event(name, 5)["text"] == ""
            and event(name, 5)["structured_content"]["phase"] == "response",
            "empty ordinary boundary or other phase changed",
        )
        require(
            event(name, 6)["text"] == records[6]["data"]["content"]
            and event(name, 6)["structured_content"]["phase"] == "detail",
            "phase metadata classified ordinary content",
        )
        mode(name)
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
                *mode(name),
            )
            page_rows.extend(page["events"])
            cursor = page["next_cursor"]
            if not cursor:
                break
        require(page_rows == rows, "Copilot eligibility skipped or repeated tied pages")
        for position, row in enumerate(rows):
            window = case.json_command(
                "show", "event", row["trwl_event_id"], "--window", "1", *mode(name)
            )
            require(
                window["events"] == rows[max(0, position - 1) : position + 2],
                "Copilot window counted excluded events",
            )
        for marker in (
            "sidecarlatebeacon",
            "summarycopilotbeacon",
            "contentcopilotbeacon",
            "arraycopilotbeacon11",
            "standalonecopilotbeacon",
        ):
            hits = case.search(
                marker, "--session", session, "--events", "--limit", "1", *mode(name)
            )["results"]
            require(
                bool(hits) == (name != "default"),
                "Copilot selected readable occurrence used the wrong query view",
            )
            if hits:
                full = case.json_command(
                    "show", "event", hits[0]["trwl_event_id"], *mode(name)
                )["event"]
                require(
                    hits[0]["snippet"] in full["text"]
                    and len(hits[0]["snippet"]) <= 400,
                    "Copilot late snippet lost whole-part provenance",
                )
        require(
            not case.search("sidecar-ciphertext", "--session", session, *mode(name))[
                "results"
            ],
            "opaque sidecar entered searchable text",
        )

    source = event("source", 2)
    archive = source["details"][1]
    require(
        source["details"][0]
        == {
            "type": "detail",
            "text": sidecar,
            "signature": "sidecar-signature",
            "data": "sidecar-ciphertext",
        },
        "source lost scalar sidecars",
    )
    require(
        archive["blocks"][:3]
        == [
            {"type": "detail", "text": duplicate, "signature": "block-signature"},
            {"type": "detail", "text": "", "data": "redacted-opaque"},
            {
                "type": "detail",
                "text": "summarycopilotbeacon Original summary.\ncontentcopilotbeacon Original native content.",
                "data": "native-ciphertext",
            },
        ],
        "source lost native block text or opaque values",
    )
    require(
        archive["orderedBlocks"][1]
        == {"type": "detail", "text": duplicate, "signature": "ordered-signature"}
        and archive["orderedBlocks"][3]
        == {
            "type": "detail",
            "text": "`protectedcopilotbeacon`",
            "signature": "protected-signature",
        },
        "source lost ordered block occurrences",
    )
    require(
        event("source", 5)["details"]
        == [
            {
                "type": "detail",
                "text": "",
                "signature": "opaqueonly-signature",
                "data": "opaqueonly-ciphertext",
            }
        ],
        "opaque-only group disappeared",
    )
    for name in ("default", "expanded"):
        require(
            "details" not in event(name, 5),
            "opaque-only group generated an empty detail",
        )
    expanded = event("expanded", 2)
    require(
        expanded["details"][1]["blocks"][0]["text"] != duplicate
        and expanded["text"].startswith(ordinary + "\n"),
        "identical ordinary and selected occurrences were globally replaced",
    )
    require(
        all(
            set(node) == {"type", "text"}
            for node in expanded["details"][1]["blocks"]
            if node["type"] == "detail"
        ),
        "expanded block retained opaque data",
    )

    for index, env in enumerate((None, "", "source", "SOURCE", " source")):
        for enabled in (False, True):
            case.env.pop("TRWL_TEXTURE", None)
            if env is not None:
                case.env["TRWL_TEXTURE"] = env
            flags = ("--texture",) if enabled else ()
            expected = event(
                "source"
                if enabled and env == "source"
                else "expanded"
                if enabled
                else "default",
                2,
            )
            for format in ("text", "markdown", "json", "jsonl"):
                args = (
                    *flags,
                    "show",
                    "event",
                    expected["trwl_event_id"],
                    "--format",
                    format,
                )
                rendered = case.command(*args).stdout
                output = case.root / f"copilot-{index}-{enabled}.{format}"
                case.command(*args, "--out", output)
                require(
                    output.read_text() == rendered, "Copilot export bypassed projection"
                )
                if format == "json":
                    value = json.loads(rendered)
                    require(
                        value["event"] == expected and value["events"] == [expected],
                        "JSON event/events disagree",
                    )
                elif format == "jsonl":
                    lines = [json.loads(line) for line in rendered.splitlines()]
                    require(
                        lines[0]["event"] == expected
                        and lines[-1]["terminal"]
                        and not lines[-1]["truncated"],
                        "JSONL projection/completion changed",
                    )
                else:
                    require(
                        expected["text"] in rendered,
                        "human output changed the chosen body",
                    )

    mode("default")
    case.command(
        "show", "event", event("source", 3)["trwl_event_id"], expect_failure=True
    )
    case.command(
        "list",
        "events",
        "--session",
        session,
        "--cursor",
        event("source", 4)["trwl_event_id"],
        "--texture",
        expect_failure=True,
    )
    case.command("import", "--provider", "copilot-cli", "--path", path)
    require(
        all(transcript(name) == rows for name, rows in views.items()),
        "repeat import changed Copilot projections",
    )
    changed = deepcopy(records)
    changed[2]["data"]["reasoningText"] = "editedsidecarbeacon New selected text."
    changed = [row for row in changed if row["id"] != identity(3)]
    write_jsonl(path, changed)
    case.command("import", "--provider", "copilot-cli", "--path", path)
    require(
        next(
            row
            for row in transcript("default")
            if row["provider_event_id"] == identity(2)
        )
        == event("default", 2),
        "selected edit changed the ordinary message",
    )
    require(
        not case.search("sidecarlatebeacon", "--session", session, *mode("expanded"))[
            "results"
        ]
        and case.search("editedsidecarbeacon", "--session", session, *mode("expanded"))[
            "results"
        ],
        "selected rewrite kept stale search text",
    )
    require(
        not any(
            row["provider_event_id"] == identity(3) for row in transcript("source")
        ),
        "deleted standalone event survived",
    )
    latest = {name: transcript(name) for name in views}
    old_data = case.env["XDG_DATA_HOME"]
    case.env["XDG_DATA_HOME"] = str(case.root / "fresh-copilot-data")
    case.command("import", "--provider", "copilot-cli", "--path", path)
    require(
        all(transcript(name) == rows for name, rows in latest.items()),
        "fresh Copilot import differs from incremental updates",
    )
    case.env["XDG_DATA_HOME"] = old_data
    path.rename(path.with_suffix(".indexed-away"))
    require(
        transcript("source") == latest["source"],
        "Copilot source retrieval depended on the source file",
    )
    mode("default")
    case.check(
        "Copilot sidecars, standalone events and mixed native replay blocks share projection, query and paging policy"
    )


def provider_details(case):
    opencode_session = "opencode-provider-details"
    opencode_path = case.providers / "provider-details/opencode/native.db"
    opencode_path.parent.mkdir(parents=True)
    opencode = OpenCodeFixture(opencode_path, case.work)
    opencode.session(opencode_session)

    opencode_control = "opencodecontrolbeacon Ordinary OpenCode message."
    opencode.message(opencode_session, "message-control", opencode_control, role="user")
    # OpenCode V1 ReasoningPart.metadata supports provider-owned nested values.
    opencode_detail = "opencodedetailbeacon Native selected part."
    opencode.message(opencode_session, "message-detail", "", role="assistant")
    opencode.part(
        opencode_session,
        "message-detail",
        "part-detail",
        {
            "type": "reasoning",
            "text": opencode_detail,
            "signature": "direct-signature",
            "metadata": {
                "anthropic": {
                    "signature": "nested-signature",
                    "redactedData": "nested-data",
                },
                "other": [None, False, 0, {}, [], "same", "same"],
            },
        },
    )

    ordinary_owned = {
        "type": "text",
        "text": "ordinarynativebeacon Original native data.",
        "metadata": {
            "anthropic": {
                "signature": "ordinary-signature",
                "redactedData": [False, 0, None],
            },
        },
    }
    opencode.part(
        opencode_session, "message-control", "part-ordinary-owned", ordinary_owned
    )
    opencode.part(
        opencode_session,
        "message-detail",
        "part-empty-owned",
        {
            "type": "reasoning",
            "text": "",
            "metadata": {
                "anthropic": {"signature": "empty-signature", "redactedData": []}
            },
        },
    )
    opencode_error = "opencodeerrorbeacon Native failure details."
    error_part = {
        "type": "tool",
        "metadata": {
            "anthropic": {"signature": "tool-signature", "redactedData": "tool value"}
        },
        "tool": "bash",
        "callID": "call_error",
        "state": {
            "status": "error",
            "input": {"command": "false"},
            "error": opencode_error,
        },
    }
    opencode.message(opencode_session, "message-error", "", role="assistant")
    opencode.part(opencode_session, "message-error", "part-error", error_part)

    patch_text = (
        "*** Begin Patch\n"
        "*** Update File: src/retry.py\n"
        "@@\n"
        "-old\n"
        "+opencodepatchbeacon Updated retry handling.\n"
        "*** End Patch"
    )
    patch_part = {
        "type": "tool",
        "tool": "apply_patch",
        "callID": "call_patch",
        "state": {
            "status": "completed",
            "input": {"patchText": patch_text},
            "output": "Updated src/retry.py",
        },
    }
    opencode.message(opencode_session, "message-patch-text", "", role="assistant")
    opencode.part(opencode_session, "message-patch-text", "part-patch-text", patch_part)

    metadata_parts = [
        (
            "unknownfilebeacon",
            {
                "type": "file",
                "text": "unknownfilebeacon Native attachment caption.",
                "filename": "attachment.bin",
                "url": "data:application/octet-stream;base64,base64onlybeacon",
                "id": "nativeidentitybeacon",
                "sessionID": "nativeidentitysessionbeacon",
            },
        ),
        (
            "opencodestartbeacon",
            {
                "type": "step-start",
                "snapshot": "opencodestartbeacon",
            },
        ),
        (
            "opencodefinishbeacon",
            {
                "type": "step-finish",
                "reason": "opencodefinishbeacon",
            },
        ),
        (
            "opencodemediapatchbeacon",
            {
                "type": "patch",
                "text": "opencodemediapatchbeacon Native patch metadata.",
                "files": ["src/metadata-change.patch"],
                "hash": "synthetic-patch-hash",
            },
        ),
        (
            "opencodeagentbeacon",
            {
                "type": "agent",
                "text": "opencodeagentbeacon Agent metadata.",
                "name": "synthetic-agent",
            },
        ),
        (
            "opencodesubtaskbeacon",
            {
                "type": "subtask",
                "text": "opencodesubtaskbeacon Subtask metadata.",
                "description": "synthetic subtask",
                "sessionID": "synthetic-subtask-session",
            },
        ),
        (
            "opencodecompactionbeacon",
            {
                "type": "compaction",
                "text": "opencodecompactionbeacon Compaction metadata.",
                "auto": True,
                "overflow": False,
                "tokens": 1234,
            },
        ),
    ]
    metadata_items = []
    for index, (marker, payload) in enumerate(metadata_parts, 1):
        message_id = f"message-metadata-{index}"
        part_id = f"part-metadata-{index}"
        opencode.message(opencode_session, message_id, "", role="assistant")
        opencode.part(opencode_session, message_id, part_id, payload)
        metadata_items.append(
            {
                "marker": marker,
                "part_id": part_id,
                "provider": "opencode",
                "session": opencode_session,
                "scope": "all",
                "text": payload.get("text", ""),
                "native": payload,
            }
        )

    large_files = []
    for index in range(128):
        component = f"directory-{index:02d}-" + "x" * 236
        large_files.append(
            "/".join(
                (
                    "synthetic",
                    component,
                    component,
                    component,
                    component,
                    f"last-file-{index:02d}.patch",
                )
            )
        )
    require(
        len(json.dumps(large_files, ensure_ascii=False).encode("utf-8")) > 65535,
        "large OpenCode patch fixture does not exceed 65535 serialized bytes",
    )
    large_patch_marker = "opencodelargepatchbeacon"
    large_patch = {
        "type": "patch",
        "text": f"{large_patch_marker} Large native patch file list.",
        "files": large_files,
        "hash": "synthetic-large-patch-hash",
    }
    large_patch_message = "message-large-patch"
    opencode.message(opencode_session, large_patch_message, "", role="assistant")
    opencode.part(
        opencode_session,
        large_patch_message,
        "part-large-patch",
        large_patch,
    )

    pi_session = "pi-provider-details"
    pi_path = case.providers / "provider-details/pi/session.jsonl"
    pi_thinking = "pithinkingbeacon Native reasoning content."
    pi_control = "picontrolbeacon Native ordinary message."
    pi_body = {
        "role": "assistant",
        "content": [
            {
                "type": "thinking",
                "thinking": pi_thinking,
                "thinkingSignature": "pi-signature",
            }
        ],
    }
    pi_control_body = {
        "role": "assistant",
        "content": [{"type": "text", "text": pi_control}],
    }
    write_jsonl(
        pi_path,
        [
            {
                "type": "session",
                "version": 3,
                "id": pi_session,
                "timestamp": "2026-10-01T00:00:00Z",
                "cwd": str(case.work),
            },
            {
                "type": "message",
                "id": "pi-thinking",
                "timestamp": "2026-10-01T00:00:01Z",
                "message": pi_body,
            },
            {
                "type": "message",
                "id": "pi-control",
                "timestamp": "2026-10-01T00:00:02Z",
                "message": pi_control_body,
            },
        ],
    )

    claude_session = "claude-provider-details"
    claude_path = case.providers / "provider-details/claude/session.jsonl"
    claude_thinking = "claudethinkingbeacon Native reasoning content."
    claude_control = "claudecontrolbeacon Native ordinary message."
    # Anthropic SDK ThinkingBlock/RedactedThinkingBlock shapes; this is not a
    # capture establishing how the Claude CLI persists these blocks.
    claude_body = {
        "role": "assistant",
        "content": [
            {
                "type": "thinking",
                "thinking": claude_thinking,
                "signature": "claude-signature",
            },
            {"type": "redacted_thinking", "data": "claude-redacted-opaque"},
        ],
    }
    claude_control_body = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": claude_control},
            {"type": "reasoning", "text": "unknownclaudebeacon Unverified form."},
            {"type": "redacted_reasoning", "data": "unknown opaque value"},
        ],
    }
    write_jsonl(
        claude_path,
        [
            {
                "type": "assistant",
                "sessionId": claude_session,
                "uuid": "claude-thinking",
                "timestamp": "2026-10-01T00:00:01Z",
                "cwd": str(case.work),
                "message": claude_body,
            },
            {
                "type": "assistant",
                "sessionId": claude_session,
                "uuid": "claude-control",
                "timestamp": "2026-10-01T00:00:02Z",
                "cwd": str(case.work),
                "message": claude_control_body,
            },
        ],
    )

    codex_session = "codex-provider-details"
    codex_path = case.providers / "provider-details/codex/session.jsonl"
    codex_reasoning = "codexreasoningbeacon Native summary-only reasoning."
    codex_control = "codexcontrolbeacon Native ordinary message."
    codex_reasoning_payload = {
        "type": "reasoning",
        "id": "codex-reasoning",
        "summary": [{"type": "summary_text", "text": codex_reasoning}],
        "content": [
            {
                "type": "reasoning_text",
                "text": "codexcontentbeacon More selected content.",
            }
        ],
        "encrypted_content": "codex-opaque",
    }
    codex_control_payload = {
        "type": "message",
        "role": "assistant",
        "id": "codex-control",
        "content": [{"type": "output_text", "text": codex_control}],
    }
    write_jsonl(
        codex_path,
        [
            {
                "type": "session_meta",
                "payload": {"id": codex_session, "cwd": str(case.work)},
            },
            {
                "type": "response_item",
                "timestamp": "2026-10-01T00:00:01Z",
                "payload": codex_reasoning_payload,
            },
            {
                "type": "response_item",
                "timestamp": "2026-10-01T00:00:02Z",
                "payload": codex_control_payload,
            },
        ],
    )

    case.command("import", "--provider", "opencode", "--path", opencode_path)
    case.command("import", "--provider", "pi", "--path", pi_path)
    case.command("import", "--provider", "claude", "--path", claude_path)
    case.command("import", "--provider", "codex", "--path", codex_path)

    checks = [
        {
            "marker": "ordinarynativebeacon",
            "provider": "opencode",
            "session": opencode_session,
            "scope": "all",
            "text": ordinary_owned["text"],
            "native": ordinary_owned,
        },
        {
            "marker": "opencodedetailbeacon",
            "selected": True,
            "provider": "opencode",
            "session": opencode_session,
            "scope": "all",
            "text": opencode_detail,
        },
        {
            "marker": "opencodeerrorbeacon",
            "provider": "opencode",
            "session": opencode_session,
            "scope": "outputs",
            "text": opencode_error,
            "native": error_part,
            "native_on_call": True,
        },
        {
            "marker": "opencodepatchbeacon",
            "provider": "opencode",
            "session": opencode_session,
            "scope": "calls",
            "text": "opencodepatchbeacon",
            "native": patch_part,
            "file": "src/retry.py",
        },
        {
            "marker": large_patch_marker,
            "provider": "opencode",
            "session": opencode_session,
            "scope": "all",
            "text": large_patch["text"],
            "native": large_patch,
            "file": large_files[-1],
        },
        *(
            item
            for item in metadata_items
            if not item["native"]["type"].startswith("step-")
        ),
        {
            "marker": "pithinkingbeacon",
            "selected": True,
            "provider": "pi",
            "session": pi_session,
            "scope": "all",
            "text": pi_thinking,
            "native": pi_body,
        },
        {
            "marker": "claudethinkingbeacon",
            "selected": True,
            "provider": "claude",
            "session": claude_session,
            "scope": "all",
            "text": claude_thinking,
            "native": claude_body,
        },
        {
            "marker": "codexreasoningbeacon",
            "selected": True,
            "provider": "codex",
            "session": codex_session,
            "scope": "all",
            "text": codex_reasoning + "\ncodexcontentbeacon More selected content.",
            "native": codex_reasoning_payload,
        },
        {
            "marker": "opencodecontrolbeacon",
            "provider": "opencode",
            "session": opencode_session,
            "scope": "transcript",
            "text": opencode_control,
            "native": {"type": "text", "text": opencode_control},
        },
        {
            "marker": "picontrolbeacon",
            "provider": "pi",
            "session": pi_session,
            "scope": "transcript",
            "text": pi_control,
            "native": pi_control_body,
        },
        {
            "marker": "claudecontrolbeacon",
            "provider": "claude",
            "session": claude_session,
            "scope": "transcript",
            "text": claude_control,
            "native": claude_control_body,
        },
        {
            "marker": "codexcontrolbeacon",
            "provider": "codex",
            "session": codex_session,
            "scope": "transcript",
            "text": codex_control,
            "native": codex_control_payload,
        },
    ]

    def capture_hit(item):
        args = [
            "--events",
            "--provider",
            item["provider"],
            "--session",
            item["session"],
            "--content-scope",
            item["scope"],
        ]
        if "file" in item:
            args.extend(("--file", item["file"]))
        if item.get("selected"):
            case.env.pop("TRWL_TEXTURE", None)
            require(
                not case.search(item["marker"], *args)["results"],
                "default indexed a selected native part",
            )
            hits = case.search(item["marker"], *args, "--texture")["results"]
            require(len(hits) == 1, "expanded query did not use original spelling")
            identity = hits[0]["trwl_event_id"]
            transformed = case.json_command("show", "event", identity, "--texture")[
                "event"
            ]
            case.command("show", "event", identity, expect_failure=True)
            case.env["TRWL_TEXTURE"] = "source"
            original = case.json_command("show", "event", identity, "--texture")[
                "event"
            ]
            require(
                original["text"] == item["text"]
                and transformed["text"] != original["text"],
                "native text mode changed",
            )
            require(
                len(transformed["text"]) == len(original["text"]),
                "selected projection changed scalar length",
            )

            def nodes(event):
                native = event["structured_content"]
                return (
                    [native]
                    if item["provider"] in ("codex", "opencode")
                    else native["content"]
                )

            require(
                nodes(transformed) == [{"type": "detail", "text": transformed["text"]}],
                "expanded native projection has a second copy or opaque metadata",
            )
            expected = [{"type": "detail", "text": item["text"]}]
            if item["provider"] == "codex":
                expected[0]["data"] = "codex-opaque"
            elif item["provider"] == "pi":
                expected[0]["signature"] = "pi-signature"
            elif item["provider"] == "opencode":
                expected[0]["signature"] = ["direct-signature", "nested-signature"]
                expected[0]["data"] = [
                    "nested-data",
                    None,
                    False,
                    0,
                    {},
                    [],
                    "same",
                    "same",
                ]
            elif item["provider"] == "claude":
                expected[0]["signature"] = "claude-signature"
                expected.append(
                    {"type": "detail", "text": "", "data": "claude-redacted-opaque"}
                )
            require(
                nodes(original) == expected,
                "source native projection lost text or opaque values",
            )
            require(
                case.search(item["marker"], *args, "--texture")["results"][0][
                    "trwl_event_id"
                ]
                == identity,
                "source query changed identity",
            )
            case.env.pop("TRWL_TEXTURE", None)
            return {
                "event_id": identity,
                "event": {"expanded": transformed, "source": original},
            }
        hits = case.search(item["marker"], *args, refresh="off")["results"]
        require(
            len(hits) == 1,
            f"{item['marker']}: expected one filtered hit, got {len(hits)}",
        )
        hit = hits[0]
        event = case.json_command("show", "event", hit["trwl_event_id"])["event"]
        require(
            event["trwl_event_id"] == hit["trwl_event_id"]
            and event["trwl_session_id"] == hit["trwl_session_id"]
            and event["provider"] == item["provider"]
            and event["provider_session_id"] == item["session"],
            f"{item['marker']}: search and show identity disagree",
        )
        require(
            item["text"] in event["text"],
            f"{item['marker']}: complete native content is not indexed",
        )
        structured = event.get("structured_content")
        if item.get("native_on_call"):
            require(structured is None, "tool output repeats the native call payload")
            neighbors = case.json_command(
                "show", "event", hit["trwl_event_id"], "--window", "1"
            )["events"]
            calls = [
                row
                for row in neighbors
                if row["event_type"] == "tool_call"
                and row["provider_event_id"] == event["provider_event_id"]
                and row["sequence"] + 1 == event["sequence"]
            ]
            require(len(calls) == 1, "tool output has no matching call record")
            structured = calls[0]["structured_content"]
        structured_text = json.dumps(structured, ensure_ascii=False)
        require(
            item["marker"] in structured_text,
            f"{item['marker']}: native structured content was not preserved",
        )
        if "native" in item:
            require(
                structured == item["native"],
                f"{item['marker']}: native structured payload changed",
            )
        return {"event_id": hit["trwl_event_id"], "event": event}

    baseline = {item["marker"]: capture_hit(item) for item in checks}
    claude_ordinary = baseline["claudecontrolbeacon"]["event"]
    for source_mode in (False, True):
        if source_mode:
            case.env["TRWL_TEXTURE"] = "source"
        require(
            case.json_command(
                "show", "event", claude_ordinary["trwl_event_id"], "--texture"
            )["event"]
            == claude_ordinary,
            "unknown Claude forms were reclassified",
        )
    case.env.pop("TRWL_TEXTURE", None)
    ordinary_event = baseline["ordinarynativebeacon"]["event"]
    for source_mode in (False, True):
        if source_mode:
            case.env["TRWL_TEXTURE"] = "source"
        require(
            case.json_command(
                "show", "event", ordinary_event["trwl_event_id"], "--texture"
            )["event"]
            == ordinary_event,
            "ordinary V1 metadata was projected",
        )
    empty_owned = next(
        row
        for row in case.json_command("show", "session", opencode_session, "--texture")[
            "events"
        ]
        if row["provider_event_id"] == "part-empty-owned"
    )
    require(
        empty_owned["structured_content"]
        == {"type": "detail", "text": "", "signature": "empty-signature", "data": []},
        "empty V1 nested values disappeared",
    )
    case.env.pop("TRWL_TEXTURE", None)
    case.command(
        "show", "event", empty_owned["trwl_event_id"], "--texture", expect_failure=True
    )
    for unsearchable in (
        "base64onlybeacon",
        "nativeidentitybeacon",
        "nativeidentitysessionbeacon",
    ):
        require(
            not case.search(unsearchable, "--events", "--provider", "opencode")[
                "results"
            ],
            f"metadata indexed an inline data URL or identifier: {unsearchable}",
        )

    opencode_metadata_rows = case.json_command(
        "list",
        "events",
        "--provider",
        "opencode",
        "--session",
        opencode_session,
    )["events"]
    metadata_events = []
    for item in metadata_items:
        rows = [
            row
            for row in opencode_metadata_rows
            if row.get("provider_event_id") == item["part_id"]
        ]
        require(
            len(rows) == 1 and rows[0].get("structured_content") == item["native"],
            f"{item['marker']}: list did not preserve its full native part",
        )
        metadata_events.extend(rows)

    input_paths = (opencode_path, pi_path, claude_path, codex_path)
    for path in input_paths:
        renamed = path.with_name(path.name + ".indexed-away")
        path.rename(renamed)
        require(not path.exists() and renamed.is_file(), f"could not hide {path}")

    for item in checks:
        after = capture_hit(item)
        before = baseline[item["marker"]]
        require(
            after["event_id"] == before["event_id"],
            f"{item['marker']}: event ID changed after input removal",
        )
        require(
            after["event"] == before["event"],
            f"{item['marker']}: show content changed after input removal",
        )
    for event in metadata_events:
        require(
            case.json_command("show", "event", event["trwl_event_id"])["event"]
            == event,
            "native metadata changed after input removal",
        )
    case.env["TRWL_TEXTURE"] = "source"
    require(
        case.json_command("show", "event", empty_owned["trwl_event_id"], "--texture")[
            "event"
        ]
        == empty_owned,
        "empty V1 native restoration depended on the source file",
    )
    case.env.pop("TRWL_TEXTURE", None)

    case.check(
        "provider details remain searchable with exact scopes and complete native show/list content after source inputs are renamed"
    )


provider_cases = {
    "pi-jsonl": pi_jsonl,
    "claude-code-jsonl": claude_code_jsonl,
    "copilot-cli-jsonl": copilot_cli_jsonl,
    "provider-details": provider_details,
}
