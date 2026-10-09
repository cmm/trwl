"""File-focused retrieval, large records, and updates to a growing JSONL history."""

import json
from collections import Counter
from pathlib import Path

from e2e_providers import copilot_records, native_records, write_jsonl
from e2e_support import OpenCodeFixture, codex_fixture, require


def benchmark_restore(case):
    from perf import scenario_spec
    from perf_semantic import PreparedSemantic

    prepared = PreparedSemantic(
        case.root / "benchmark",
        case.binary,
        case.work,
        scenario_spec("semantic-index", "opencode", "long", 2, 256),
        30,
        None,
    )
    try:
        original = prepared.case.json_command(
            "list", "events", "--workspace", case.work
        )["events"]
        for _ in range(2):
            prepared.corpus.append()
            prepared.case.command("import", "--all")
            prepared.restore()
            require(
                prepared.case.json_command("list", "events", "--workspace", case.work)[
                    "events"
                ]
                == original,
                "restoration changed prepared history",
            )
            report = prepared.case.json_command("import", "--all")
            require(
                report["sources"]
                and all(row.get("unchanged") for row in report["sources"]),
                "restored source would trigger import during query timing",
            )
    finally:
        prepared.close(True)
    case.check(
        "benchmark restoration preserves history and leaves source stamps current"
    )


def discovery(case):
    require(
        case.json_command("status")["data_root"] == str(case.data),
        "TRWL_DATA_ROOT did not select the data directory",
    )
    override = case.root / "other-data"
    require(
        case.json_command("--data-root", override, "status")["data_root"]
        == str(override),
        "--data-root did not override TRWL_DATA_ROOT",
    )
    data_root = case.env.pop("TRWL_DATA_ROOT")
    data_home = case.env["XDG_DATA_HOME"]
    try:
        require(
            case.json_command("status")["data_root"] == str(Path(data_home) / "trwl"),
            "XDG data directory did not use the trwl name",
        )
        del case.env["XDG_DATA_HOME"]
        require(
            case.json_command("status")["data_root"]
            == str(case.home / ".local/share/trwl"),
            "home data directory did not use the trwl name",
        )
    finally:
        case.env["TRWL_DATA_ROOT"] = data_root
        case.env["XDG_DATA_HOME"] = data_home
    case.check("data-root flag, environment override, XDG default and home default")

    database = Path(case.env["XDG_DATA_HOME"]) / "opencode/opencode.db"
    database.parent.mkdir(parents=True)
    fixture = OpenCodeFixture(database, case.work)
    fixture.session("ses_discovered")
    opencode_text = (
        "discoveropencodebeacon A conversation in the default OpenCode location."
    )
    fixture.message("ses_discovered", "msg_discovered", opencode_text)
    codex_session = "550e8400-e29b-41d4-a716-446655440033"
    codex_text = "discovercodexbeacon A conversation in the configured Codex home."
    codex_fixture(
        Path(case.env["CODEX_HOME"]) / "sessions" / f"rollout-{codex_session}.jsonl",
        case.work,
        codex_session,
        [("user", codex_text)],
    )
    discovered = [
        ("discoveropencodebeacon", opencode_text, "opencode", "ses_discovered"),
        ("discovercodexbeacon", codex_text, "codex", codex_session),
    ]
    for provider, root in (
        ("pi", case.home / ".pi/agent/sessions"),
        ("claude", Path(case.env["CLAUDE_CONFIG_DIR"]) / "projects"),
    ):
        session = f"{provider}-discovered"
        records, expected = native_records(provider, case.work, session)
        write_jsonl(root / "-workspace" / f"{session}.jsonl", records)
        marker, body, _, _ = expected[0]
        discovered.append((marker, body, provider, session))
    copilot_session = "copilot-discovered"
    records, expected = copilot_records(case.work, copilot_session)
    write_jsonl(
        case.home / ".copilot/session-state" / copilot_session / "events.jsonl", records
    )
    marker, body, _, _ = expected[0]
    discovered.append((marker, body, "copilot_cli", copilot_session))
    config = case.data / "config.toml"
    config.write_text(
        config.read_text()
        .replace("automatic = false", "automatic = true")
        .replace('mode = "manual"', 'mode = "auto"')
    )
    identities = {}
    try:
        for _ in range(2):
            case.command("setup", "--wait", timeout=120)
            for marker, body, provider, session in discovered:
                hits = case.search(marker, "--events")["results"]
                require(len(hits) == 1, "setup missed or duplicated discovered history")
                hit = hits[0]
                event = case.json_command("show", "event", hit["trwl_event_id"])[
                    "event"
                ]
                require(
                    event["text"] == body
                    and event["provider"] == provider
                    and event["provider_session_id"] == session,
                    "discovered history has the wrong content or conversation",
                )
                identity = (event["trwl_event_id"], event["trwl_session_id"])
                require(
                    identities.setdefault(marker, identity) == identity,
                    "repeated setup changed the discovered history's references",
                )
    finally:
        # Setup starts maintenance itself; stop it through the public command.
        case.command("index", "mode", "manual", timeout=20)
    case.check(
        "setup discovers all five coding harnesses; repeating setup preserves content and references"
    )


def file_search(case):
    session = "550e8400-e29b-41d4-a716-446655440031"
    path = codex_fixture(
        case.providers / "files.jsonl",
        case.work,
        session,
        [("user", "fileworkflow Merely mentioning src/retry.rs in conversation.")],
    )
    for name in ("retry", "other"):
        patch = (
            f"*** Begin Patch\n*** Add File: src/{name}.rs\n"
            f"+// fileworkflow {name} change\n*** End Patch"
        )
        with path.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "timestamp": "2026-09-21T12:00:02Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "apply_patch",
                            "call_id": f"call_{name}",
                            "arguments": patch,
                        },
                    }
                )
                + "\n"
            )
    case.command("import", "--provider", "codex", "--path", path)
    all_hits = case.search("fileworkflow", "--events")["results"]
    require(len(all_hits) == 3, "fixture lost the message or either file operation")
    hits = case.search("fileworkflow", "--events", "--file", "src/retry.rs")["results"]
    require(len(hits) == 1, "file filter included prose or the other file operation")
    event = case.json_command("show", "event", hits[0]["trwl_event_id"])["event"]
    require(
        event["provider_session_id"] == session
        and event["event_type"] == "tool_call"
        and "*** Add File: src/retry.rs" in event["text"],
        "file search returned the wrong operation",
    )
    require(
        not case.search("fileworkflow", "--events", "--file", "src/absent.rs")[
            "results"
        ],
        "unknown file filter returned unrelated history",
    )
    case.check(
        "file search selects the recorded operation among matching counterexamples"
    )

    fixture = OpenCodeFixture(case.providers / "terms.db", case.work)
    fixture.session("ses_query_terms")
    fixture.message(
        "ses_query_terms", "msg_first", "firstalternative Reconnect the client."
    )
    fixture.message(
        "ses_query_terms", "msg_second", "secondalternative Retry the request."
    )
    case.command("import", "--provider", "opencode", "--path", fixture.path)
    hits = case.search("firstalternative", "--term", "secondalternative", "--events")[
        "results"
    ]
    texts = [
        case.json_command("show", "event", hit["trwl_event_id"])["event"]["text"]
        for hit in hits
    ]
    require(
        Counter(texts)
        == Counter(
            [
                "firstalternative Reconnect the client.",
                "secondalternative Retry the request.",
            ]
        ),
        "alternative terms did not find both complete messages exactly once",
    )
    case.check("alternative search terms broaden retrieval without duplicates")
    require(
        {
            hit["trwl_event_id"]
            for hit in case.search("firstalternative secondalternative", "--events")[
                "results"
            ]
        }
        == {hit["trwl_event_id"] for hit in hits},
        "plain query words did not act as alternatives",
    )
    require(
        not case.search("+firstalternative +secondalternative", "--events")["results"],
        "required words matched messages that contain only one",
    )
    require(
        len(case.search('"Reconnect the client"', "--events")["results"]) == 1,
        "quoted phrase did not match its full word order",
    )
    require(
        not case.search('"client the Reconnect"', "--events")["results"],
        "quoted phrase ignored word order",
    )
    case.check("query alternatives, required words and phrases preserve selection")


def long_content(case):
    fixture = OpenCodeFixture(case.providers / "long.db", case.work)
    session = "ses_long_content"
    fixture.session(session)
    middle = "résumé 日本語 λ → result = {'ok': True}\n" * 2048
    text = (
        "messageheadbeacon Start of the message.\n" + middle + "messagetailbeacon End."
    )
    output = (
        "outputheadbeacon Start of the tool output.\n"
        + middle
        + "outputtailbeacon End."
    )
    fixture.message(session, "msg_long", text)
    fixture.part(
        session,
        "msg_long",
        "part_output",
        {"type": "tool_result", "callID": "call_long", "text": output},
    )
    case.command("import", "--provider", "opencode", "--path", fixture.path)
    for marker, body, kind in (
        ("messagetailbeacon", text, "message"),
        ("outputtailbeacon", output, "tool_output"),
    ):
        hits = case.search(marker, "--events", "--event-type", kind)["results"]
        require(len(hits) == 1, f"search omitted or duplicated the end of the {kind}")
        require(marker in hits[0]["snippet"], "lexical snippet omitted the late match")
        event = case.json_command("show", "event", hits[0]["trwl_event_id"])["event"]
        require(
            event["text"] == body and event["provider_session_id"] == session,
            f"retrieval truncated or changed the full {kind}",
        )
    case.check(
        "message and tool-output tails are searchable; full Unicode bodies survive retrieval"
    )

    fixture.path.rename(case.providers / "archived-input.db")
    for marker, body in (("messagetailbeacon", text), ("outputtailbeacon", output)):
        hits = case.search(marker, "--events")["results"]
        require(len(hits) == 1, "stored history disappeared with the original input")
        event = case.json_command("show", "event", hits[0]["trwl_event_id"])["event"]
        require(
            event["text"] == body,
            "retrieval needed the original input to recover complete content",
        )
    case.check(
        "indexed content remains searchable and retrievable after the original input moves away"
    )


def jsonl_updates(case):
    session = "550e8400-e29b-41d4-a716-446655440032"
    base = "stablejsonlbeacon The unchanged conversation context."
    old = "oldjsonlbeacon The answer before correction."
    new = old.replace("oldjsonlbeacon", "newjsonlbeacon")
    added = "appendedjsonlbeacon A message written after the initial import."
    path = codex_fixture(
        case.providers / "updates.jsonl",
        case.work,
        session,
        [("user", base), ("assistant", old)],
    )
    args = ("import", "--provider", "codex", "--path", path)

    def verify(expected, previous=None):
        rows = case.json_command("list", "events", "--provider", "codex")["events"]
        require(
            Counter(row["text"] for row in rows) == Counter(expected),
            "JSONL update lost, duplicated, or changed message content",
        )
        identities = {
            row["text"]: (row["trwl_event_id"], row["trwl_session_id"]) for row in rows
        }
        require(
            len({identity[0] for identity in identities.values()}) == len(rows),
            "distinct JSONL messages share an event ID",
        )
        require(
            all(row["provider_session_id"] == session for row in rows),
            "JSONL update changed the native conversation identity",
        )
        if previous:
            for body, identity in previous.items():
                require(
                    identities.get(body) == identity,
                    "unchanged JSONL message IDs changed",
                )
        return identities

    case.command(*args)
    initial = verify([base, old])
    record = (
        json.dumps(
            {
                "timestamp": "2026-09-21T12:00:01Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": added}],
                },
            }
        )
        + "\n"
    )
    split = len(record) // 2
    with path.open("a") as stream:
        stream.write(record[:split])
    result = case.command(*args, expect_failure=None)
    require(result.returncode >= 0, "partial JSONL input crashed the import process")
    verify([base, old], initial)
    with path.open("a") as stream:
        stream.write(record[split:])
    case.command(*args)
    verify([base, old, added], initial)
    case.check(
        "a partially written JSONL record preserves history and appears once when completed"
    )

    length = path.stat().st_size
    path.write_text(path.read_text().replace(old, new))
    require(path.stat().st_size == length, "fixture rewrite changed file length")
    case.command(*args)
    after_edit = verify([base, new, added], {base: initial[base]})
    require(
        not case.search("oldjsonlbeacon", "--events")["results"],
        "rewritten JSONL content remains searchable",
    )
    require(
        len(case.search("newjsonlbeacon", "--events")["results"]) == 1,
        "rewritten JSONL content is missing or duplicated in search",
    )
    case.check("a same-length JSONL edit replaces searchable content")

    codex_fixture(path, case.work, session, [("user", base), ("assistant", new)])
    for _ in range(2):
        case.command(*args)
        verify([base, new], {body: after_edit[body] for body in (base, new)})
        require(
            not case.search("appendedjsonlbeacon", "--events")["results"],
            "removed JSONL message remains searchable",
        )
    case.check(
        "JSONL truncation and an unchanged repeat preserve surviving messages without duplicates"
    )
