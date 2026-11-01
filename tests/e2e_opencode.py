"""OpenCode v1/v2 coexistence and updates through the CLI.

The v2 fixture uses the columns consumed from upstream's session/sql.ts and
the JSON shapes in schema/src/session-message.ts at commit
a22754f761f868a80231859c4b8be8887f81187c of anomalyco/opencode.
Separate parts follow OpenCode 2.2.5's session/sql.ts and history.ts withParts.
It is hand-built, not a recording from a running OpenCode v2 instance.
"""

import json
import sqlite3

from e2e_support import OpenCodeFixture, require


def opencode_v2(case):
    fixture = OpenCodeFixture(case.providers / "mixed.db", case.work)
    fixture.session("ses_v1")
    fixture.message("ses_v1", "msg_v1", "legacybeacon Existing v1 history.")
    case.json_command("import", "--provider", "opencode", "--path", fixture.path)
    legacy = case.json_command("show", "session", "ses_v1")["events"]

    with sqlite3.connect(fixture.path) as db:
        db.executescript("""
            CREATE TABLE session_v2 (id TEXT PRIMARY KEY, directory TEXT NOT NULL,
                parent_id TEXT);
            CREATE TABLE session_message (id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES session_v2(id) ON DELETE CASCADE,
                type TEXT NOT NULL, seq INTEGER NOT NULL,
                time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL,
                data TEXT NOT NULL);
            CREATE UNIQUE INDEX session_message_session_seq_idx
                ON session_message(session_id, seq);
        """)
        db.execute(
            "INSERT INTO session_v2 VALUES (?, ?, NULL)",
            ("ses_v2", str(case.work)),
        )

    def message(message_id, kind, data, seq, session="ses_v2"):
        stamp = fixture.stamp()
        data = {"time": {"created": stamp}, **data}
        with sqlite3.connect(fixture.path) as db:
            db.execute(
                "INSERT INTO session_message VALUES (?, ?, ?, ?, ?, ?, ?)",
                (message_id, session, kind, seq, stamp, stamp, json.dumps(data)),
            )

    message("msg_user", "user", {"text": "v2beacon Search the new history."}, 1)
    case.json_command("import", "--provider", "opencode", "--path", fixture.path)
    hits = case.search("v2beacon", "--events")["results"]
    require(len(hits) == 1, "v2 message missing from a database containing v1 and v2")
    user = case.json_command("show", "event", hits[0]["trwl_event_id"])["event"]
    require(
        user["text"] == "v2beacon Search the new history."
        and user["role"] == "user"
        and user["provider_session_id"] == "ses_v2"
        and user["provider_message_id"] == "msg_user",
        "v2 user content or identity lost",
    )
    require(
        case.json_command("show", "session", "ses_v1")["events"] == legacy,
        "v2 import changed v1 history",
    )
    case.check(
        "v2 messages become searchable while v1 history keeps its content and IDs"
    )

    # V2's seq orders messages even when IDs, insertion order, and tied
    # timestamps would otherwise put this inserted message after msg_user.
    message("msg_z_early", "user", {"text": "earliestbeacon Inserted earlier."}, 0)
    with sqlite3.connect(fixture.path) as db:
        db.execute(
            "UPDATE session_message SET time_created = "
            "(SELECT time_created FROM session_message WHERE id = 'msg_user') "
            "WHERE id = 'msg_z_early'"
        )

    # Growing tool state changes the number of trwl events in the middle of a
    # session; later records must be resequenced, not left stale or duplicated.
    assistant = {
        "tokens": {"input": 8, "output": 5, "reasoning": 3},
        "error": {
            "_tag": "UnknownError",
            "message": "assistantfailurebeacon Original error.",
        },
        "content": [
            {
                "type": "reasoning",
                "text": "reasonbeacon Inspect the failing request.",
                "native": {
                    "provider": {
                        "nested": [
                            {"signature": "v2-signature"},
                            {"redactedData": "v2-data"},
                            False,
                            0,
                            None,
                            {},
                            [],
                            "same",
                            "same",
                        ]
                    }
                },
            },
            {
                "type": "text",
                "text": "answerbeacon I will inspect the file.",
                "native": {
                    "provider": {
                        "signature": "ordinary-signature",
                        "reasoning": "ordinary native value",
                    }
                },
            },
            {
                "type": "tool",
                "id": "call_read",
                "name": "read",
                "native": {
                    "provider": {
                        "signature": "tool-signature",
                        "reasoning": "tool native value",
                    }
                },
                "state": {"status": "running", "input": {"filePath": "src/main.rs"}},
            },
            {
                "type": "tool",
                "id": "call_bash",
                "name": "bash",
                "state": {"status": "streaming", "input": '{"command":"ls'},
            },
            {
                "type": "reasoning",
                "text": "",
                "native": {"provider": {"signature": "empty-signature", "payload": []}},
            },
        ],
    }
    message("msg_assistant", "assistant", assistant, 2)
    for n in range(70):
        message(f"msg_tail_{n:02}", "user", {"text": f"Tail message {n}."}, n + 3)

    def imported():
        case.json_command("import", "--provider", "opencode", "--path", fixture.path)
        return case.json_command("show", "session", "ses_v2")["events"]

    before = imported()
    require(
        [e["provider_event_id"] for e in before[:2]] == ["msg_z_early", "msg_user"],
        "v2 seq ordering was lost at tied timestamps",
    )
    calls = [e for e in before if e["event_type"] == "tool_call"]
    require(len(calls) == 2, "v2 assistant tools were not separated into calls")
    require(any("src/main.rs" in e["files"] for e in calls), "v2 tool file paths lost")
    require(
        not case.search("reasonbeacon", "--events")["results"],
        "matched selected thinking",
    )
    detail_id = next(
        e["trwl_event_id"]
        for e in before
        if e["provider_event_id"] == "msg_assistant" and e["event_type"] == "message"
    )
    ordinary = case.json_command("show", "event", detail_id)["event"]
    require(
        ordinary["text"]
        == "answerbeacon I will inspect the file.\nUnknownError\nassistantfailurebeacon Original error.",
        "redaction lost ordinary content or appended v2 error",
    )
    require(
        ordinary["structured_content"]["content"] == assistant["content"][1:-1],
        "redaction changed ordinary V2 native siblings",
    )
    require(
        ordinary["structured_content"]["tokens"]
        == {"input": 8, "output": 5, "detail_tokens": 3},
        "v2 counter metadata changed",
    )
    for secret in ("reasonbeacon", "v2-signature", "v2-data", "empty-signature"):
        require(
            secret not in json.dumps(ordinary),
            f"V2 retrieval exposed thinking payload: {secret}",
        )
    assistant["content"][2]["state"] = {
        "status": "completed",
        "input": {"filePath": "src/main.rs"},
        "content": [
            {"type": "text", "text": "outputbeacon Complete file.\nSecond line."}
        ],
    }
    assistant["content"][3]["state"] = {
        "status": "error",
        "input": {"command": "ls missing"},
        "error": {"_tag": "UnknownError", "message": "errorbeacon Missing directory."},
    }
    # Insert content before both tools: IDs must follow calls, not array offsets.
    assistant["content"].insert(
        0, {"type": "text", "text": "prefacebeacon New preface."}
    )
    with sqlite3.connect(fixture.path) as db:
        db.execute(
            "UPDATE session_message SET data = ?, time_updated = ? WHERE id = ?",
            (json.dumps(assistant), fixture.stamp(), "msg_assistant"),
        )
    after = imported()
    require(
        {e["trwl_event_id"] for e in after if e["event_type"] == "tool_call"}
        == {e["trwl_event_id"] for e in calls},
        "streaming or content insertion changed tool-call IDs",
    )
    for marker, body in [
        ("outputbeacon", "outputbeacon Complete file.\nSecond line."),
        ("errorbeacon", "errorbeacon Missing directory."),
    ]:
        hits = case.search(marker, "--events", "--content-scope", "outputs")["results"]
        require(len(hits) == 1, f"expected one v2 tool output for {marker}")
        event = case.json_command("show", "event", hits[0]["trwl_event_id"])["event"]
        require(body in event["text"], "tool completion lost output/error text")
        require(
            event["provider_event_id"] == "msg_assistant",
            "tool location lost its row ID",
        )
    require(
        [e["sequence"] for e in after] == list(range(len(after))),
        "tool completion left gaps or duplicated suffix sequences",
    )
    require(imported() == after, "unchanged v2 import changed the transcript")
    case.check(
        "streaming tools complete with stable IDs, searchable output/errors, and a resequenced suffix"
    )

    with sqlite3.connect(fixture.path) as db:
        db.execute(
            "INSERT INTO session_v2 VALUES (?, ?, ?)",
            ("ses_child", str(case.work), "ses_v2"),
        )
    message("msg_child", "user", {"text": "childbeacon Child work."}, 1, "ses_child")
    message(
        "msg_compact",
        "compaction",
        {
            "status": "completed",
            "summary": "summarybeacon Summary.",
            "tokens": {"input": 8, "output": 5, "reasoning": 3},
        },
        80,
    )
    message("msg_system", "system", {"text": "systembeacon Updated instructions."}, 81)
    message(
        "msg_shell",
        "shell",
        {
            "command": "printf shellcallbeacon",
            "output": {"output": "shelloutbeacon Done."},
        },
        82,
    )
    message(
        "msg_compact_empty",
        "compaction",
        {"status": "completed", "reason": "auto", "summary": "", "native": {"messages": []}},
        83,
    )
    message("msg_after_compact", "user", {"text": "aftercompactbeacon Continue work."}, 84)
    attempts = [
        ("running", {"status": "running", "summary": "runningcompactbeacon Partial summary."}),
        ("unknown", {"status": "unknown", "summary": "unknowncompactbeacon Partial summary."}),
        ("missing", {"summary": "missingcompactbeacon Partial summary."}),
        ("failed", {"status": "failed", "error": {
            "_tag": "UnknownError", "message": "failedcompactbeacon Compaction was not applied."
        }}),
    ]
    for index, (outcome, data) in enumerate(attempts, 85):
        message(f"msg_compact_{outcome}", "compaction", data, index)
    with_compactions = imported()
    compact = next(e for e in with_compactions if e["provider_event_id"] == "msg_compact")
    empty_compact = next(
        e for e in with_compactions if e["provider_event_id"] == "msg_compact_empty"
    )
    require(
        compact["event_type"] == empty_compact["event_type"] == "compaction"
        and compact["text"] == "summarybeacon Summary."
        and empty_compact["text"] == ""
        and empty_compact["structured_content"]["native"] == {"messages": []},
        "v2 summary or text-free compaction boundary lost its kind or native data",
    )
    require(
        compact["structured_content"]["tokens"]
        == {"input": 8, "output": 5, "detail_tokens": 3},
        "v2 compaction exposed category-only token metadata",
    )
    require(
        [
            hit["trwl_event_id"]
            for hit in case.search("summarybeacon", "--events", "--event-type", "compaction")["results"]
        ] == [compact["trwl_event_id"]],
        "v2 compaction summary is not searchable as a compaction event",
    )
    require(imported() == with_compactions, "unchanged v2 import changed compactions")
    case.check("v2 summaries and text-free compaction boundaries keep their native content and kind")
    require(
        [
            event["trwl_event_id"]
            for event in case.json_command(
                "list", "events", "--session", "ses_v2", "--event-type", "compaction"
            )["events"]
        ] == [compact["trwl_event_id"], empty_compact["trwl_event_id"]],
        "v2 unsuccessful or incomplete compaction became a boundary",
    )
    for outcome, data in attempts:
        event = next(
            event for event in with_compactions
            if event["provider_event_id"] == f"msg_compact_{outcome}"
        )
        body = data.get("summary", data.get("error", {}).get("message"))
        require(
            event["event_type"] == "metadata"
            and body in event["text"]
            and all(event["structured_content"][key] == value for key, value in data.items()),
            f"v2 {outcome} compaction lost its metadata or native content",
        )
        require(
            [
                hit["trwl_event_id"] for hit in case.search(
                    body.split()[0], "--events", "--session", "ses_v2", "--event-type", "metadata"
                )["results"]
            ] == [event["trwl_event_id"]],
            f"v2 {outcome} compaction text is not searchable as metadata",
        )
    require(
        case.search(
            "aftercompactbeacon", "--events", "--session", "ses_v2", "--after-compaction", "latest"
        )["results"],
        "v2 latest boundary selection used a later unsuccessful compaction",
    )
    case.check("v2 running, failed, unknown-status and status-free compactions remain metadata and do not advance latest")
    child = case.json_command("show", "session", "ses_child")["events"][0]
    require(
        child["parent_trwl_session_id"] == user["trwl_session_id"],
        "v2 parent relationship lost",
    )
    for marker in (
        "summarybeacon",
        "systembeacon",
        "shellcallbeacon",
        "shelloutbeacon",
    ):
        require(case.search(marker, "--events")["results"], f"missing {marker}")

    with sqlite3.connect(fixture.path) as db:
        db.execute("DELETE FROM session_message WHERE id = 'msg_tail_00'")
        db.execute("DELETE FROM session_v2 WHERE id = 'ses_child'")
        db.execute(
            "UPDATE session_message SET data = 'broken', time_updated = ? WHERE id = 'msg_system'",
            (fixture.stamp(),),
        )
    after = imported()
    require(
        not case.search("childbeacon", "--events")["results"],
        "deleted v2 session remains searchable",
    )
    require(
        not case.search("systembeacon", "--events")["results"],
        "rejected v2 row kept stale content",
    )
    require(
        not any(e["provider_event_id"] == "msg_tail_00" for e in after),
        "deleted v2 message survived",
    )
    require(
        [e["sequence"] for e in after] == list(range(len(after))),
        "deletion broke suffix sequences",
    )
    case.check(
        "v2 metadata, shell output, children, rejection, and deletion affect retrieval"
    )

    # OpenCode migrates into session_v2 while keeping the v1 rows and session ID.
    with sqlite3.connect(fixture.path) as db:
        db.execute(
            "INSERT INTO session_v2 VALUES (?, ?, NULL)", ("ses_v1", str(case.work))
        )
    message(
        "msg_migrated",
        "user",
        {"text": "legacybeacon Existing v1 history."},
        1,
        "ses_v1",
    )
    imported()
    migrated = case.json_command("show", "session", "ses_v1")["events"]
    require(
        len(migrated) == 1
        and migrated[0]["trwl_session_id"] == legacy[0]["trwl_session_id"]
        and migrated[0]["provider_event_id"] == "msg_migrated",
        "migration duplicated the transcript or lost session identity",
    )
    with sqlite3.connect(fixture.path) as db:
        db.execute("DELETE FROM session_message WHERE session_id = 'ses_v1'")
    imported()
    require(
        not case.search("legacybeacon", "--events")["results"],
        "empty v2 transcript resurrected v1 rows",
    )
    case.check(
        "migrated v2 sessions replace legacy copies, including empty transcripts"
    )

    # A v2-only database must not need empty legacy tables to parse successfully.
    with sqlite3.connect(fixture.path) as db:
        for table in ("part", "message", "session"):
            db.execute(f"DROP TABLE {table}")
    require(imported() == after, "dropping legacy tables changed v2 history")
    case.env["XDG_DATA_HOME"] = str(case.root / "fresh-data")
    require(imported() == after, "incremental v2 output differs from a fresh import")
    case.check(
        "v2-only and mixed databases agree; incremental results match a fresh import"
    )

    # OpenCode 2.2.5 session/history.ts withParts restores data.content from
    # session_part in ordinal order, falling back to inline content without parts.
    with sqlite3.connect(fixture.path) as db:
        db.executescript("""
            CREATE TABLE session_part (
                message_id TEXT NOT NULL REFERENCES session_message(id) ON DELETE CASCADE,
                session_id TEXT NOT NULL REFERENCES session_v2(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL, type TEXT NOT NULL, call_id TEXT,
                time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL,
                data TEXT NOT NULL, PRIMARY KEY (message_id, ordinal));
        """)
        for ordinal, part in reversed(list(enumerate(assistant["content"]))):
            stamp = fixture.stamp()
            db.execute(
                "INSERT INTO session_part VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "msg_assistant",
                    "ses_v2",
                    ordinal * 10,
                    part["type"],
                    part.get("id"),
                    stamp,
                    stamp,
                    json.dumps(part),
                ),
            )
        db.execute(
            "UPDATE session_message SET data = ?, time_updated = ? WHERE id = 'msg_assistant'",
            (json.dumps({**assistant, "content": []}), fixture.stamp()),
        )
    separate = imported()
    require(separate == after, "separate v2 parts lost content, order, or event IDs")
    require(
        not case.search("reasonbeacon", "--events")["results"],
        "separate v2 parts made thinking searchable",
    )
    for secret in ("reasonbeacon", "v2-signature", "v2-data", "empty-signature"):
        require(secret not in json.dumps(separate),
                f"separate V2 parts exposed thinking payload: {secret}")

    message(
        "msg_inline",
        "assistant",
        {"content": [{"type": "text", "text": "inlinebeacon Inline fallback."}]},
        90,
    )
    imported()
    require(
        case.search("inlinebeacon", "--events")["results"],
        "part table suppressed an assistant without separate parts",
    )

    with sqlite3.connect(fixture.path) as db:
        db.execute(
            "UPDATE session_message SET data = ?, time_updated = ? WHERE id = 'msg_assistant'",
            (
                json.dumps(
                    {
                        **assistant,
                        "content": [
                            {
                                "type": "text",
                                "text": "stalebeacon Stale inline content.",
                            }
                        ],
                    }
                ),
                fixture.stamp(),
            ),
        )
    imported()
    require(
        not case.search("stalebeacon", "--events")["results"],
        "inline content overrode separate parts",
    )
    case.json_command(
        "sources", "add", "parts", "--provider", "opencode", "--root", fixture.path
    )

    # Only child rows change: parent message stamps and payload stay untouched.
    replacement = {"type": "text", "text": "revisedbeacon Revised answer."}
    tool = {
        **assistant["content"][3],
        "state": {
            "status": "completed",
            "input": {"filePath": "src/main.rs"},
            "content": [{"type": "text", "text": "revisedoutputbeacon Revised file."}],
        },
    }
    with sqlite3.connect(fixture.path) as db:
        for ordinal, part in [(20, replacement), (30, tool)]:
            db.execute(
                "UPDATE session_part SET data = ?, time_updated = ? "
                "WHERE message_id = 'msg_assistant' AND ordinal = ?",
                (json.dumps(part), fixture.stamp(), ordinal),
            )
    require(
        case.search("revisedbeacon", "--events", refresh="wait")["results"],
        "refresh missed part-only edits",
    )
    revised = case.json_command("show", "session", "ses_v2")["events"]
    require(
        not case.search("answerbeacon", "--events")["results"],
        "part-only edit kept old assistant text",
    )
    require(
        not case.search("outputbeacon", "--events", "--content-scope", "outputs")[
            "results"
        ],
        "part-only edit kept old tool output",
    )
    require(
        case.search("revisedoutputbeacon", "--events", "--content-scope", "outputs")[
            "results"
        ],
        "part-only edit missed new tool output",
    )
    require(
        {
            e["trwl_event_id"]
            for e in revised
            if e["event_type"] == "tool_call"
            and e["provider_event_id"] == "msg_assistant"
        }
        == {e["trwl_event_id"] for e in calls},
        "part-only edit changed tool-call identities",
    )

    with sqlite3.connect(fixture.path) as db:
        stamp = fixture.stamp()
        db.execute(
            "INSERT INTO session_part VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "msg_assistant",
                "ses_v2",
                5,
                "text",
                None,
                stamp,
                stamp,
                json.dumps({"type": "text", "text": "insertbeacon Inserted text."}),
            ),
        )
        db.execute(
            "DELETE FROM session_part WHERE message_id = 'msg_assistant' AND ordinal = 30"
        )
    require(
        case.search("insertbeacon", "--events", refresh="wait")["results"],
        "part-only insertion was missed",
    )
    changed = case.json_command("show", "session", "ses_v2")["events"]
    require(
        not case.search(
            "revisedoutputbeacon", "--events", "--content-scope", "outputs"
        )["results"],
        "deleted part left tool output searchable",
    )
    require(
        len(
            [
                e
                for e in changed
                if e["event_type"] == "tool_call"
                and e["provider_event_id"] == "msg_assistant"
            ]
        )
        == 1,
        "deleted part left a tool call",
    )
    require(
        [e["sequence"] for e in changed] == list(range(len(changed))),
        "part insertion/deletion broke suffix sequences",
    )
    require(imported() == changed, "unchanged separate parts changed the transcript")
    case.env["XDG_DATA_HOME"] = str(case.root / "parts-fresh-data")
    require(
        imported() == changed, "incremental separate parts differ from fresh import"
    )
    with sqlite3.connect(fixture.path) as db:
        db.execute("DELETE FROM session_part WHERE message_id = 'msg_assistant'")
    imported()
    require(
        not case.search("revisedbeacon", "--events")["results"]
        and not case.search("insertbeacon", "--events")["results"],
        "deleting all separate parts left stale assistant content",
    )
    require(
        case.search("stalebeacon", "--events")["results"],
        "deleting all parts did not restore inline fallback",
    )
    case.check(
        "separate v2 parts preserve content and IDs; child edits, insertions, and deletions refresh"
    )
