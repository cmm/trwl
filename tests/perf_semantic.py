"""Embedding throughput and queries use separate prepared states per executable."""

import json
import shutil
import time
import traceback

from e2e_support import require, write_json
from perf_fixtures import Corpus, verify_corpus
from perf_support import PerformanceCase

QUERY_WORKLOADS = ("semantic-search", "hybrid-search", "semantic-append")
QUERY = "How should request handlers manage caches and timeouts?"


def search_args(backend, workspace):
    return (
        "search",
        QUERY,
        "--workspace",
        str(workspace),
        "--backend",
        backend,
        "--refresh",
        "wait",
        "--events",
        "--limit",
        "3",
        "--format",
        "json",
    )


def verify_semantic(case, corpus, identities, result, backend):
    value = json.loads(result.stdout)
    require(
        value["retrieval"]["effective_mode"] == backend
        and value["retrieval"]["semantic_status"] == "ready",
        "semantic benchmark did not use the requested backend",
    )
    require(0 < len(value["results"]) <= 3, "semantic results are empty or unbounded")
    known = {identity[0] for identity in identities.values()}
    for hit in value["results"]:
        require(hit["trwl_event_id"] in known, "semantic result has an unknown event")
        shown = json.loads(
            case.command(
                "show", "event", hit["trwl_event_id"], "--format", "json"
            ).stdout
        )["event"]
        require(
            shown["text"] in corpus.expected, "semantic result lost recorded content"
        )


class PreparedSemantic:
    def __init__(self, root, binary, workspace, spec, timeout, model_dir):
        self.case = PerformanceCase(root, binary, model_dir)
        self.profile = spec["profile"]
        self.timeout = timeout
        self.snapshot = root / "prepared-data"
        self.before_source = root / "before.db"
        self.after_source = root / "after.db"
        self.preparation = {"result": "failed", "measurements": {}}
        started = time.monotonic()
        try:
            self.prepare(workspace, spec)
            self.preparation["result"] = "passed"
        except BaseException:
            self.preparation["traceback"] = traceback.format_exc()
            self.case.close()
            raise
        finally:
            self.preparation["elapsed_seconds"] = time.monotonic() - started
            write_json(root / "preparation.json", self.preparation)

    def prepare(self, workspace, spec):
        case = self.case
        self.corpus = Corpus(
            case.providers / "input",
            workspace,
            "opencode",
            "long",
            spec["size"],
            message_bytes=spec["message_bytes"],
            long_message_bytes=spec["long_message_bytes"],
        )
        self.before = self.corpus.manifest()
        case.command(
            "sources",
            "add",
            "perf",
            "--provider",
            "opencode",
            "--root",
            self.corpus.path,
        )
        case.command("import", "--all", "--progress", "none", timeout=self.timeout)
        self.identities = verify_corpus(case, self.corpus)
        config = case.config / "config.toml"
        config.write_text(
            config.read_text() + "[semantic]\nbuiltin_throttling = false\n"
        )
        case.command(
            "semantic", "enable", "--executor", "builtin", timeout=self.timeout
        )
        if self.profile == "semantic-query":
            result, measured = case.measured(
                "prepare-embeddings",
                *search_args("semantic", self.corpus.workspace),
                timeout=self.timeout,
            )
            self.preparation["measurements"]["prepare-embeddings"] = measured
            verify_semantic(case, self.corpus, self.identities, result, "semantic")
        # Snapshots belong to this executable and keep the same absolute source paths.
        # Copying and source restoration are outside every measured operation.
        if case.runtime.exists():
            case.command("daemon", "stop", timeout=self.timeout)
        for name in ("data", "config", "cache"):
            path = getattr(case, name)
            if path.exists():
                shutil.copytree(path, self.snapshot / name)
        shutil.copy2(self.corpus.path, self.before_source)
        if self.profile == "semantic-query":
            self.corpus.append()
            self.after = self.corpus.manifest()
            shutil.copy2(self.corpus.path, self.after_source)
        else:
            self.after = self.before
        self.preparation.update(before=self.before, after=self.after)
        self.preparation["query_input"] = {
            "directories": {
                name: str(getattr(case, name))
                for name in ("data", "config", "cache", "runtime")
            },
            "query": QUERY,
            "workspace": str(self.corpus.workspace),
            "corpus": self.after,
        }

    def restore(self):
        if self.case.runtime.exists():
            self.case.command("daemon", "stop", timeout=self.timeout)
        for name in ("data", "config", "cache"):
            path = getattr(self.case, name)
            if path.exists():
                shutil.rmtree(path)
            if (self.snapshot / name).exists():
                shutil.copytree(self.snapshot / name, path)
        shutil.copy2(self.before_source, self.corpus.path)
        # copy2 restores mtime, but ctime changes. Reconcile the source stamp
        # before timing queries against unchanged history.
        self.case.command("import", "--all", "--progress", "none", timeout=self.timeout)
        status = self.case.json_command("index", "status")
        require(status["state"] == "ready", "restored lexical index is not ready")

    def trial(self):
        case = self.case
        report = {
            "result": "failed",
            "measurements": {},
            "before": self.before,
            "after": self.after,
        }
        try:
            self.restore()
            if self.profile == "semantic-index":
                result, measured = case.measured(
                    "semantic-index",
                    *search_args("semantic", self.corpus.workspace),
                    timeout=self.timeout,
                )
                report["measurements"]["semantic-index"] = measured
                verify_semantic(case, self.corpus, self.identities, result, "semantic")
            else:
                for backend in ("semantic", "hybrid"):
                    workload = backend + "-search"
                    result, measured = case.measured(
                        workload,
                        *search_args(backend, self.corpus.workspace),
                        timeout=self.timeout,
                    )
                    report["measurements"][workload] = measured
                    verify_semantic(case, self.corpus, self.identities, result, backend)
                shutil.copy2(self.after_source, self.corpus.path)
                result, measured = case.measured(
                    "semantic-append",
                    *search_args("semantic", self.corpus.workspace),
                    timeout=self.timeout,
                )
                report["measurements"]["semantic-append"] = measured
                identities = verify_corpus(case, self.corpus, self.identities)
                verify_semantic(case, self.corpus, identities, result, "semantic")
            report["result"] = "passed"
        except Exception as exc:  # noqa: BLE001 - report failed samples without losing earlier pairs
            report.update(error=str(exc), traceback=traceback.format_exc())
        return report

    def close(self, keep_data):
        self.case.close()
        if not keep_data:
            for name in ("data", "config", "cache"):
                path = getattr(self.case, name)
                if path.exists():
                    shutil.rmtree(path)
            shutil.rmtree(self.snapshot)
