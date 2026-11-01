"""Compaction boundaries delimit indexed events independently of timestamps."""

import json

from e2e_http import EmbeddingServer
from e2e_support import require


def compaction_selection(case):
    marker = "segmentbeacon"
    root = case.providers / "compaction"
    root.mkdir()
    stamp = "2026-09-22T12:00:00Z"

    def message(name, timestamp=stamp):
        return {
            "type": "response_item",
            "timestamp": timestamp,
            "payload": {
                "type": "message",
                "id": name,
                "role": "user",
                "content": [{"type": "input_text", "text": f"{marker} {name}"}],
            },
        }

    def boundary(name, timestamp=stamp):
        return {
            "type": "compacted",
            "timestamp": timestamp,
            "payload": {"message": name},
        }

    rows = {
        "first": [
            message("first-old"),
            boundary("first-cut-one"),
            message("first-middle"),
            # Source order, not the timestamp, defines the latest boundary.
            boundary("first-cut-two", "2026-09-20T12:00:00Z"),
            message("first-new"),
        ],
        "second": [
            message("second-old"),
            boundary("second-cut"),
            message("second-new"),
        ],
        "uncompacted": [message("uncompacted-only")],
    }
    for session, events in rows.items():
        header = {
            "type": "session_meta",
            "payload": {"id": session, "cwd": str(case.work)},
        }
        (root / f"rollout-{session}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in [header, *events])
        )
    case.json_command("import", "--provider", "codex", "--path", root)
    events = case.json_command("list", "events", "--limit", "100")["events"]
    cuts = {
        event["text"]: event["trwl_event_id"]
        for event in events
        if event["event_type"] == "compaction"
    }
    require(
        set(cuts) == {"first-cut-one", "first-cut-two", "second-cut"},
        f"missing compaction boundaries: {cuts}",
    )
    ids = {
        event["provider_event_id"]: event["trwl_event_id"]
        for event in events
        if event["event_type"] == "message"
    }

    def check(expected, *filters, backend="lexical"):
        report = case.search(
            marker, "--events", "--limit", "100", *filters, backend=backend
        )
        require(
            report["retrieval"]["effective_mode"] == backend,
            f"{backend} selection check used a different backend: {report['retrieval']}",
        )
        actual = {hit["trwl_event_id"] for hit in report["results"]}
        require(
            actual == {ids[name] for name in expected},
            f"{backend} {filters}: expected {expected}, got {actual}",
        )

    after = ("--after-compaction", "latest")
    before = ("--before-compaction", "latest")
    between = (
        "--after-compaction",
        cuts["first-cut-one"],
        "--before-compaction",
        cuts["first-cut-two"],
    )
    check({"first-new", "second-new"}, *after)
    check({"first-old", "first-middle", "second-old"}, *before)
    check({"first-middle"}, *between)
    check({"first-old"}, "--before-compaction", cuts["first-cut-one"])
    check(
        {"first-middle", "first-new"},
        "--after-compaction",
        cuts["first-cut-one"],
    )
    check(
        {"first-middle"},
        "--after-compaction",
        cuts["first-cut-one"],
        *before,
    )
    check(set(), *after, *before)
    check(set(), *after, "--session", "uncompacted")
    check(set(), "--after-compaction", cuts["first-cut-one"], "--session", "second")
    check({"first-new"}, *after, "--session", "first")
    check({"second-new"}, *after, "--exclude-session", "first")
    check(set(), *after, "--provider", "pi")
    check(
        {"first-new", "second-new"},
        *after,
        "--since",
        "2026-09-22",
        "--event-type",
        "message",
        "--content-scope",
        "transcript",
    )
    case.check(
        "before/after use transcript order and resolve boundaries before content/date filters"
    )

    listed = case.json_command("list", "events", *between)["events"]
    require(
        [event["trwl_event_id"] for event in listed] == [ids["first-middle"]],
        "list did not apply the same exclusive boundary range",
    )
    listed = case.json_command("list", "events", "--event-type", "compaction", *after)[
        "events"
    ]
    require(not listed, "after latest included a boundary itself")
    grouped = case.search(marker, *after)["results"]
    require(
        {hit["trwl_session_id"] for hit in grouped}
        == {
            event["trwl_session_id"]
            for event in events
            if event["provider_event_id"] in ("first-new", "second-new")
        },
        "session grouping changed compaction selection",
    )
    for anchor, detail in [
        ("missing-event", "compaction event not found"),
        (ids["first-old"], "event is not a compaction boundary"),
    ]:
        result = case.command(
            "search",
            marker,
            "--refresh",
            "off",
            "--after-compaction",
            anchor,
            expect_failure=True,
        )
        require(
            detail in result.stderr,
            f"wrong invalid-boundary diagnostic: {result.stderr}",
        )
    case.check("list, session grouping, exclusive cuts, and invalid boundary IDs")

    # Equal vectors make every message a semantic candidate, so boundary filters
    # must exclude out-of-range anchors in each backend.
    with EmbeddingServer(embed=lambda texts: [[1.0, 0.0] for _ in texts]) as server:
        server.enable(case)
        case.command("semantic", "rebuild", timeout=120)
        for backend in ("semantic", "hybrid"):
            check({"first-new", "second-new"}, *after, backend=backend)
            check({"first-old", "first-middle", "second-old"}, *before, backend=backend)
            check({"first-middle"}, *between, backend=backend)
    case.check("semantic and hybrid results obey the same compaction ranges")

    path = root / "rollout-first.jsonl"
    with path.open("a") as stream:
        for event in [boundary("first-cut-three"), message("first-newest")]:
            stream.write(json.dumps(event) + "\n")
    case.json_command("import", "--provider", "codex", "--path", root)
    events = case.json_command("list", "events", "--limit", "100")["events"]
    ids.update(
        (event["provider_event_id"], event["trwl_event_id"])
        for event in events
        if event["event_type"] == "message"
    )
    check({"first-newest", "second-new"}, *after)
    check({"first-middle"}, *between)
    case.check(
        "appending a compaction advances latest while explicit ranges remain stable"
    )
