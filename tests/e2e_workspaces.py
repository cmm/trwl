"""Workspace identity and selection through lexical, semantic and hybrid CLI reads."""

import json
import shutil

from e2e_http import EmbeddingServer
from e2e_providers import native_records, write_jsonl
from e2e_support import OpenCodeFixture, codex_fixture, git, require


def workspaces(case):
    marker = "workspaceorchard"
    main = case.work
    child = main / "src"
    nested = main / "nested"
    peer = case.root / "workspace-peer"
    linked = case.root / "linked"
    absent = case.root / "missing"
    alias = case.root / "alias"
    for path in (child, nested, peer):
        path.mkdir(parents=True)
    git(main, "init", "--quiet")
    git(main, "commit", "--quiet", "--allow-empty", "-m", "fixture")
    git(main, "worktree", "add", "--quiet", "--detach", linked)
    git(nested, "init", "--quiet")
    git(peer, "init", "--quiet", "--object-format=sha256")
    (peer / "src").mkdir()
    alias.symlink_to(main, target_is_directory=True)

    fixture = OpenCodeFixture(case.providers / "workspaces.db", child)
    for session, directory in (
        ("local-open", child),
        ("nested", nested),
        ("peer", peer),
        ("linked", linked),
        ("unresolved", absent),
    ):
        fixture.workspace = directory
        fixture.session(session)
        fixture.message(session, session + "-message", marker + " " + session)
    case.json_command("import", "--provider", "opencode", "--path", fixture.path)
    codex = codex_fixture(
        case.providers / "codex.jsonl",
        alias / "src",
        "local-codex",
        [("user", marker + " codex")],
    )
    case.json_command("import", "--provider", "codex", "--path", codex)
    for provider in ("pi", "claude"):
        records, _ = native_records(provider, child, "local-" + provider)
        for record in records:
            if record.get("message", {}).get("role") == "user":
                record["message"]["content"][0]["text"] = marker + " " + provider
                break
        path = case.providers / (provider + ".jsonl")
        write_jsonl(path, records)
        case.json_command("import", "--provider", provider, "--path", path)
    copilot = case.providers / "events.jsonl"
    write_jsonl(
        copilot,
        [
            {
                "type": "session.start",
                "data": {"sessionId": "local-copilot", "context": {"cwd": str(child)}},
            },
            {
                "type": "user.message",
                "id": "copilot-user",
                "data": {"content": marker + " copilot"},
            },
        ],
    )
    case.json_command("import", "--provider", "copilot-cli", "--path", copilot)
    local = {"local-open", "local-codex", "local-pi", "local-claude", "local-copilot"}
    all_sessions = local | {"nested", "peer", "linked", "unresolved"}

    baseline = case.search(marker, "--workspace", "**", "--limit", "100")["results"]
    for hit in baseline:
        session = hit["provider_session_id"]
        expected = (
            None
            if session == "unresolved"
            else str(
                main
                if session in local
                else {"nested": nested, "peer": peer, "linked": linked}[session]
            )
        )
        event = case.json_command("show", "event", hit["trwl_event_id"])["event"]
        location = case.json_command("locate", "event", hit["trwl_event_id"])
        require(
            hit["workspace_root"]
            == event["workspace_root"]
            == location["workspace_root"]
            == expected,
            f"wrong canonical root for {session}",
        )
        if session == "local-codex":
            require(
                event["workspace"] == str(alias / "src"),
                "original provider directory lost",
            )
    case.check(
        "all providers preserve original directories and store canonical worktree roots"
    )

    selections = [
        ((), local),
        (("--workspace", main), local),
        (("--workspace", alias / "src"), local),
        (("--workspace", "**/workspace"), local),
        (("--workspace", str(main) + "/*"), {"nested"}),
        (("--workspace", str(main) + "/**"), {"nested"}),
        (("--workspace", str(case.root) + "/*"), local | {"peer", "linked"}),
        (("--workspace", str(case.root) + "/**"), all_sessions - {"unresolved"}),
        (
            ("--workspace", str(case.root) + "/{workspace,workspace-peer}"),
            local | {"peer"},
        ),
        (("--workspace", main, "--workspace", peer), local | {"peer"}),
        (("--workspace", str(case.root) + "/workspac[ep]-pee?"), {"peer"}),
        (("--workspace", "*"), set()),
        (("--workspace", "**/WORKSPACE"), local),
        (("--workspace", "**/absent"), set()),
        (("--workspace", "**"), all_sessions),
    ]
    with EmbeddingServer(embed=lambda texts: [[1.0, 0.0] for _ in texts]) as api:
        api.enable(case)
        for backend in ("lexical", "semantic", "hybrid"):
            for mode in ((), ("--events",)):
                for args, expected in selections:
                    result = case.search(
                        marker,
                        *mode,
                        *args,
                        "--limit",
                        "100",
                        backend=backend,
                        refresh="wait",
                    )
                    require(
                        result["retrieval"]["effective_mode"] == backend,
                        "retriever fell back",
                    )
                    actual = {hit["provider_session_id"] for hit in result["results"]}
                    require(
                        actual == expected,
                        f"{backend} {mode} {args}: {actual} != {expected}",
                    )
            case.check(
                backend + " workspace selection agrees in event and session modes"
            )
        for args, expected in selections:
            for kind in ("events", "sessions"):
                rows = case.json_command("list", kind, *args, "--limit", "100")[kind]
                require(
                    {row["provider_session_id"] for row in rows} == expected,
                    f"list {kind} workspace selection differs: {args}",
                )
        case.check("list events and sessions use the same default and glob selection")

        for cwd, expected in (
            (child, local),
            (alias / "src", local),
            (nested, {"nested"}),
            (linked, {"linked"}),
        ):
            case.work = cwd
            for backend in ("lexical", "semantic", "hybrid"):
                actual = {
                    hit["provider_session_id"]
                    for hit in case.search(marker, "--limit", "100", backend=backend)[
                        "results"
                    ]
                }
                require(actual == expected, f"wrong default from {cwd}: {actual}")
        case.work = main
        case.check(
            "current subdirectories, symlinks, nested repositories and linked worktrees select their roots"
        )

        empty = case.root / "empty"
        empty.mkdir()
        git(empty, "init", "--quiet")
        case.work = empty
        require(
            not case.search(marker)["results"],
            "unindexed current workspace broadened search",
        )
        case.work = main
        for pattern in ("[", ""):
            failure = case.command(
                "search", marker, "--workspace", pattern, expect_failure=True
            )
            require(
                "workspace" in failure.stderr.lower(),
                "invalid pattern lacks a diagnostic",
            )
        case.check("unknown current workspace stays empty and malformed patterns fail")

        # A changed symlink must not reassign old sessions when an input appends.
        alias.unlink()
        alias.symlink_to(peer, target_is_directory=True)
        rows = [json.loads(line) for line in codex.read_text().splitlines()]
        rows.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": marker + " appended"}],
                },
            }
        )
        write_jsonl(codex, rows)
        case.json_command("import", "--provider", "codex", "--path", codex)
        new = codex_fixture(
            case.providers / "new.jsonl",
            alias / "src",
            "new-alias",
            [("user", marker + " new session")],
        )
        case.json_command("import", "--provider", "codex", "--path", new)
        for hit in case.search(
            marker, "--workspace", "**", "--events", "--limit", "100"
        )["results"]:
            if hit["provider_session_id"] in ("local-codex", "new-alias"):
                expected = main if hit["provider_session_id"] == "local-codex" else peer
                require(
                    hit["workspace_root"] == str(expected),
                    "session root changed after symlink retarget",
                )
        case.check(
            "incremental import keeps old roots and resolves new sessions at their new locations"
        )

        archived = case.root / "archived"
        archived.mkdir()
        git(archived, "init", "--quiet")
        archive_input = codex_fixture(
            case.providers / "archive.jsonl",
            archived,
            "archived",
            [("user", marker + " archived")],
        )
        case.json_command("import", "--provider", "codex", "--path", archive_input)
        shutil.rmtree(archived)
        case.command("import", "--provider", "codex", "--path", archive_input, "--full")
        hits = case.search(marker, "--workspace", archived)["results"]
        require(
            len(hits) == 1 and hits[0]["workspace_root"] == str(archived),
            "reparse lost the canonical root of a deleted checkout",
        )
        require(
            not case.search(marker, "--session", hits[0]["trwl_session_id"])["results"],
            "session filter bypassed workspace restriction",
        )
        require(
            case.json_command("show", "session", hits[0]["trwl_session_id"])["events"],
            "explicit transcript retrieval depends on the current workspace",
        )
        case.check(
            "deleted checkouts retain roots and direct transcript retrieval remains global"
        )
