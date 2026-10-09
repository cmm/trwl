"""Native coding-harness histories, checked through the public CLI."""

import json

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


def provider_details(case):
    opencode_session = "opencode-provider-details"
    opencode_path = case.providers / "provider-details/opencode/native.db"
    opencode_path.parent.mkdir(parents=True)
    opencode = OpenCodeFixture(opencode_path, case.work)
    opencode.session(opencode_session)

    opencode_control = "opencodecontrolbeacon Ordinary OpenCode message."
    opencode.message(opencode_session, "message-control", opencode_control, role="user")

    opencode_error = "opencodeerrorbeacon Native failure details."
    error_part = {
        "type": "tool",
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
        "content": [{"type": "thinking", "thinking": pi_thinking}],
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
    claude_body = {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": claude_thinking}],
    }
    claude_control_body = {
        "role": "assistant",
        "content": [{"type": "text", "text": claude_control}],
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
        "content": [],
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
            "provider": "pi",
            "session": pi_session,
            "scope": "all",
            "text": pi_thinking,
            "native": pi_body,
        },
        {
            "marker": "claudethinkingbeacon",
            "provider": "claude",
            "session": claude_session,
            "scope": "all",
            "text": claude_thinking,
            "native": claude_body,
        },
        {
            "marker": "codexreasoningbeacon",
            "provider": "codex",
            "session": codex_session,
            "scope": "all",
            "text": codex_reasoning,
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

    case.check(
        "provider details remain searchable with exact scopes and complete native show/list content after source inputs are renamed"
    )


provider_cases = {
    "pi-jsonl": pi_jsonl,
    "claude-code-jsonl": claude_code_jsonl,
    "copilot-cli-jsonl": copilot_cli_jsonl,
    "provider-details": provider_details,
}
