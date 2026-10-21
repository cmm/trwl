#!/usr/bin/env python3
"""Check a requested GPU's real-model embeddings against a CPU baseline."""

import argparse
import json
import math
import os
import signal
import sqlite3
import struct
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

from e2e_semantic import QUERY, _fixture
from e2e_support import SCRATCH, Case, require


class Interrupted(BaseException):
    pass


def interrupted(signum, _frame):
    # InterruptedError can be retried by subprocess; unwind instead.
    raise Interrupted(f"interrupted by signal {signum}")


def require_execution(directory, provider):
    profiles = sorted(directory.glob("*.json"))
    require(profiles, f"no ONNX execution profiles in {directory}")
    executed = set()
    for path in profiles:
        for event in json.loads(path.read_text()):
            args = event.get("args", {})
            # Registration, session initialization and transfer/fence events do
            # not demonstrate that a model operation ran on the provider.
            if (
                event.get("cat") == "Node"
                and event.get("ph") == "X"
                and event.get("name", "").endswith("_kernel_time")
                and args.get("op_name")
                and not args["op_name"].startswith("Memcpy")
                and args.get("provider")
            ):
                executed.add(args["provider"])
    require(
        provider in executed,
        f"no model operation executed by {provider}; "
        f"profiled providers: {sorted(executed)} ({directory})",
    )


def run_case(root, mode, args):
    case = Case(root / mode, args.binary, args.model_dir)
    try:
        for name, path in (
            ("ORT_DYLIB_PATH", args.ort_dylib),
            ("OCL_ICD_VENDORS", args.opencl_vendors),
        ):
            if path is not None:
                case.env[name] = str(path)
        if mode == "cpu":
            case.env["TRWL_SEMANTIC_CPU_ONLY"] = "1"
        profiles = case.root / "profiles"
        profiles.mkdir()
        case.env["TRWL_SEMANTIC_PROFILE_DIR"] = str(profiles)
        _, expected = _fixture(case)
        config = case.data / "config.toml"
        config.write_text(
            config.read_text() + "[semantic]\nbuiltin_throttling = false\n"
        )
        case.command("semantic", "enable", "--executor", "builtin", timeout=180)
        built = case.command("semantic", "rebuild", "--format", "json", timeout=240)
        require(json.loads(built.stdout)["state"] == "ready", "rebuild was not ready")
        if mode == "gpu":
            prefix = "trwl: embedding update using "
            devices = [
                line.removeprefix(prefix)
                for line in built.stderr.splitlines()
                if line.startswith(prefix)
            ]
            require(
                devices == [args.device],
                f"requested {args.device}, rebuild reported {devices}: "
                f"{case.commands[-1].log}",
            )
            require(
                "finishing on CPU" not in built.stderr,
                f"GPU inference fell back to CPU: {case.commands[-1].log}",
            )
        found = case.search(
            QUERY,
            "--events",
            "--limit",
            "3",
            backend="semantic",
            refresh="off",
            timeout=180,
        )
        require(
            found["retrieval"]["effective_mode"] == "semantic"
            and found["retrieval"]["semantic_status"] == "ready",
            f"{mode} semantic retrieval was not ready",
        )
        hits = found["results"]
        require(
            1 <= len(hits) <= 3
            and any(hit["trwl_event_id"] == expected["trwl_event_id"] for hit in hits),
            f"{mode} paraphrase missed the target among the top three results",
        )
        require_execution(
            profiles,
            "CPUExecutionProvider"
            if mode == "cpu"
            else {
                "cuda": "CUDAExecutionProvider",
                "openvino-gpu": "OpenVINOExecutionProvider",
            }[args.device],
        )
        database = case.data / "semantic/cache.sqlite3"
        vectors = {}
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            for key, blob in connection.execute("SELECT key, vector FROM vectors"):
                require(
                    len(blob) > 0 and len(blob) % 4 == 0,
                    f"{mode} invalid float vector: {key}",
                )
                vectors[key] = struct.unpack(f"<{len(blob) // 4}f", blob)
        return vectors
    finally:
        case.close()


def compare_vectors(cpu, gpu):
    require(cpu.keys() == gpu.keys(), "CPU/GPU embedding key sets differ")
    require(len(gpu) >= 9, "missing fixture document embeddings (need at least nine)")
    dimensions = {len(vector) for vector in cpu.values()}
    require(
        len(dimensions) == 1 and next(iter(dimensions)) > 0,
        "CPU embedding dimensions are empty or inconsistent",
    )
    minimum_cosine = float("inf")
    for key, vector in gpu.items():
        baseline = cpu[key]
        require(len(vector) == len(baseline), f"CPU/GPU dimensions differ: {key}")
        require(all(map(math.isfinite, baseline)), f"nonfinite CPU embedding: {key}")
        require(all(map(math.isfinite, vector)), f"nonfinite GPU embedding: {key}")
        require(
            abs(math.fsum(value * value for value in vector) - 1) <= 1e-5,
            f"GPU embedding is not normalized: {key}",
        )
        cosine = math.fsum(a * b for a, b in zip(baseline, vector))
        minimum_cosine = min(minimum_cosine, cosine)
        require(cosine > 0.999, f"GPU embedding differs from CPU: {key}: {cosine}")
    return minimum_cosine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True, help="trwl executable")
    parser.add_argument(
        "--model-dir", type=Path, help="Model directory (default: compiled-in model)"
    )
    parser.add_argument("--device", choices=("cuda", "openvino-gpu"), required=True)
    parser.add_argument(
        "--ort-dylib", type=Path, help="Set ORT_DYLIB_PATH in both isolated cases"
    )
    parser.add_argument(
        "--opencl-vendors", type=Path, help="Set OCL_ICD_VENDORS in both isolated cases"
    )
    args = parser.parse_args()
    scratch = SCRATCH / "assistant/tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="trwl-gpu-", dir=scratch))
    print(f"Artifacts: {root}", flush=True)
    handlers = {
        sig: signal.signal(sig, interrupted)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        for name in ("binary", "model_dir", "ort_dylib", "opencl_vendors"):
            path = getattr(args, name)
            if path is not None:
                path = path.resolve(strict=True)
                setattr(args, name, path)
        require(
            args.binary.is_file() and os.access(args.binary, os.X_OK),
            f"--binary must name an executable file: {args.binary}",
        )
        cpu = run_case(root, "cpu", args)
        gpu = run_case(root, "gpu", args)
        minimum_cosine = compare_vectors(cpu, gpu)
    except BaseException as exc:
        print(f"{args.device}: failed — {exc}", file=sys.stderr)
        return 1
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    print(
        f"{args.device}: passed; {len(gpu)} vectors, "
        f"minimum CPU/GPU cosine {minimum_cosine:.8f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
