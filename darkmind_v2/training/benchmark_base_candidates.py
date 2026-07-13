"""Isolated BF16 microbenchmarks for DarkMind v2 Phase 3A candidates."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from darkmind_v2.modeling.configuration_darkmind_v2 import DarkMindV2Config
from darkmind_v2.modeling.model_io import load_model_package, save_model_package
from darkmind_v2.modeling.modeling_darkmind_v2 import DarkMindV2ForCausalLM


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "darkmind_v2" / "config"
REPORT_DIR = ROOT / "darkmind_v2" / "reports"
RUNTIME_ROOT = ROOT / "darkmind_v2" / "data" / "phase3a" / "benchmark"
CANDIDATE_PATHS = {
    "A": CONFIG_DIR / "model_base_candidate_a_60m.json",
    "B": CONFIG_DIR / "model_base_candidate_b_80m.json",
    "C": CONFIG_DIR / "model_base_candidate_c_100m.json",
    "D": CONFIG_DIR / "model_base_candidate_d_120m.json",
}
ATTENTION_PATHS = ("sdpa", "fallback")
CHECKPOINTING_VALUES = (True, False)
MIN_SAFE_HEADROOM_PERCENT = 15.0
MIN_BATCH_TWO_HEADROOM_PERCENT = 25.0


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _tensor_bytes(tensors: list[torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _optimizer_bytes(optimizer: torch.optim.Optimizer) -> int:
    tensors: list[torch.Tensor] = []
    for state in optimizer.state.values():
        tensors.extend(value for value in state.values() if torch.is_tensor(value))
    return _tensor_bytes(tensors)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _base_result(args: argparse.Namespace, config: DarkMindV2Config) -> dict[str, Any]:
    return {
        "schema_version": "darkmind-v2-phase3a-benchmark-profile-v1",
        "candidate": args.candidate,
        "config_path": CANDIDATE_PATHS[args.candidate].relative_to(ROOT).as_posix(),
        "parameters": None,
        "precision": "bf16",
        "sequence_length": config.block_size,
        "micro_batch_size": args.batch_size,
        "gradient_checkpointing": args.gradient_checkpointing == "on",
        "attention_implementation": args.attention,
        "warmup_microsteps": args.warmup,
        "measured_microsteps": args.steps,
        "deterministic_seed": args.seed,
        "checkpoint_benchmark_requested": args.checkpoint,
        "forward_success": False,
        "backward_success": False,
        "optimizer_step_success": False,
        "loss_finite": False,
        "gradients_finite": False,
        "oom": False,
        "cuda_error": None,
        "hard_failure": None,
    }


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    raw_config = json.loads(CANDIDATE_PATHS[args.candidate].read_text(encoding="utf-8"))
    raw_config["attention_implementation"] = args.attention
    raw_config["gradient_checkpointing"] = args.gradient_checkpointing == "on"
    config = DarkMindV2Config(**raw_config)
    result = _base_result(args, config)
    if not torch.cuda.is_available():
        result["hard_failure"] = "CUDA is unavailable"
        return result

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    result.update(
        {
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
        }
    )

    try:
        model = DarkMindV2ForCausalLM(config).to(device=device, dtype=torch.bfloat16)
        model.train()
        result["parameters"] = model.parameter_count()
        result["tied_embeddings"] = model.embeddings_are_tied()
        model_memory = _tensor_bytes([parameter for parameter in model.parameters()])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, betas=(0.9, 0.95), foreach=False)
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + args.batch_size)
        input_ids = torch.randint(
            8,
            config.vocab_size,
            (args.batch_size, config.block_size),
            dtype=torch.long,
            device=device,
            generator=generator,
        )

        last_loss = float("nan")

        def microstep() -> float:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(input_ids, labels=input_ids)
                loss = output.loss
            if loss is None:
                raise RuntimeError("model did not return a training loss")
            result["forward_success"] = True
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite loss")
            loss.backward()
            result["backward_success"] = True
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            result["optimizer_step_success"] = True
            return float(loss.detach())

        for _ in range(args.warmup):
            last_loss = microstep()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        durations: list[float] = []
        losses: list[float] = []
        for _ in range(args.steps):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            last_loss = microstep()
            torch.cuda.synchronize(device)
            durations.append(time.perf_counter() - started)
            losses.append(last_loss)

        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        gradients_finite = all(bool(torch.all(torch.isfinite(gradient))) for gradient in gradients)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        gradient_memory = _tensor_bytes(gradients)
        optimizer_memory = _optimizer_bytes(optimizer)
        total_duration = sum(durations)
        total_tokens = args.steps * args.batch_size * config.block_size
        result.update(
            {
                "loss_finite": all(math.isfinite(value) for value in losses),
                "gradients_finite": gradients_finite,
                "last_loss": last_loss,
                "tokens_per_second": total_tokens / total_duration,
                "samples_per_second": args.steps * args.batch_size / total_duration,
                "average_microstep_seconds": statistics.fmean(durations),
                "p50_microstep_seconds": statistics.median(durations),
                "p95_microstep_seconds": _percentile(durations, 0.95),
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "model_memory_bytes": model_memory,
                "optimizer_state_memory_bytes": optimizer_memory,
                "gradient_memory_bytes": gradient_memory,
                "activation_memory_estimate_bytes": max(
                    0, peak_allocated - model_memory - optimizer_memory - gradient_memory
                ),
                "vram_headroom_percent": (properties.total_memory - peak_reserved) / properties.total_memory * 100,
                "safe_vram_margin": (
                    (properties.total_memory - peak_reserved) / properties.total_memory * 100
                    >= MIN_SAFE_HEADROOM_PERCENT
                ),
            }
        )
        if not result["loss_finite"] or not gradients_finite:
            result["hard_failure"] = "numerical instability"
        elif not result["safe_vram_margin"]:
            result["hard_failure"] = "unsafe VRAM margin"

        if args.checkpoint and result["hard_failure"] is None:
            profile_name = _profile_name(
                args.candidate,
                args.batch_size,
                args.gradient_checkpointing == "on",
                args.attention,
            )
            checkpoint_dir = args.run_dir / f"{profile_name}_model_checkpoint"
            torch.cuda.synchronize(device)
            checkpoint_started = time.perf_counter()
            save_model_package(model, checkpoint_dir)
            torch.cuda.synchronize(device)
            checkpoint_save_seconds = time.perf_counter() - checkpoint_started
            checkpoint_bytes = sum(path.stat().st_size for path in checkpoint_dir.rglob("*") if path.is_file())
            reload_started = time.perf_counter()
            reloaded = load_model_package(checkpoint_dir, device=str(device))
            torch.cuda.synchronize(device)
            checkpoint_reload_seconds = time.perf_counter() - reload_started
            result.update(
                {
                    "checkpoint_save_success": True,
                    "checkpoint_reload_success": (
                        reloaded.parameter_count() == model.parameter_count() and reloaded.embeddings_are_tied()
                    ),
                    "checkpoint_save_seconds": checkpoint_save_seconds,
                    "checkpoint_reload_seconds": checkpoint_reload_seconds,
                    "checkpoint_file_bytes": checkpoint_bytes,
                    "checkpoint_path": checkpoint_dir.relative_to(ROOT).as_posix(),
                }
            )
    except torch.cuda.OutOfMemoryError as exc:
        result["oom"] = True
        result["hard_failure"] = "CUDA out of memory"
        result["cuda_error"] = str(exc)
        torch.cuda.empty_cache()
    except RuntimeError as exc:
        message = str(exc)
        result["cuda_error"] = message
        if "out of memory" in message.lower():
            result["oom"] = True
            result["hard_failure"] = "CUDA out of memory"
            torch.cuda.empty_cache()
        else:
            result["hard_failure"] = f"runtime error: {message}"
    except Exception as exc:  # benchmark failures must be recorded, not silently omitted
        result["hard_failure"] = f"{type(exc).__name__}: {exc}"
    return result


def _profile_name(candidate: str, batch: int, checkpointing: bool, attention: str) -> str:
    gc_name = "gc_on" if checkpointing else "gc_off"
    return f"candidate_{candidate.lower()}_b{batch}_{gc_name}_{attention}"


def _warning_labels(stderr: str) -> list[str]:
    labels: list[str] = []
    lowered = stderr.lower()
    checks = (
        ("not compiled with flash attention", "flash_attention_unavailable"),
        ("cublas_workspace_config", "cublas_determinism_warning"),
        ("memory efficient attention defaults to a non-deterministic", "sdpa_backend_nondeterminism_warning"),
        ("torch.cpu.amp.autocast", "torch_checkpoint_autocast_deprecation"),
    )
    for marker, label in checks:
        if marker in lowered:
            labels.append(label)
    return labels


def sanitize_benchmark_report(payload: dict[str, Any]) -> dict[str, Any]:
    observed: set[str] = set()
    for profile in payload.get("profiles", []):
        stderr = profile.get("worker_stderr") or ""
        labels = _warning_labels(stderr)
        observed.update(labels)
        profile["worker_warnings"] = labels
        if profile.get("worker_exit_code") == 0:
            profile["worker_stderr"] = None
        elif stderr:
            profile["worker_stderr"] = "worker stderr retained only in ignored runtime artifacts"
    environment = payload.setdefault("environment", {})
    environment["warning_labels"] = sorted(observed)
    environment["determinism_note"] = (
        "Seeds, inputs, ordering, and protocol were fixed. The measured Windows PyTorch backend warned that "
        "CuBLAS and memory-efficient attention may not be bitwise deterministic."
    )
    environment["sdpa_note"] = (
        "PyTorch SDPA was measured, but this build reported that Flash Attention was unavailable."
    )
    return payload


def _run_profile(
    *,
    run_dir: Path,
    candidate: str,
    batch_size: int,
    checkpointing: bool,
    attention: str,
    warmup: int,
    steps: int,
    seed: int,
    checkpoint: bool,
) -> dict[str, Any]:
    name = _profile_name(candidate, batch_size, checkpointing, attention)
    output = run_dir / f"{name}.json"
    command = [
        sys.executable,
        "-m",
        "darkmind_v2.training.benchmark_base_candidates",
        "--worker",
        "--candidate",
        candidate,
        "--batch-size",
        str(batch_size),
        "--gradient-checkpointing",
        "on" if checkpointing else "off",
        "--attention",
        attention,
        "--warmup",
        str(warmup),
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--run-dir",
        str(run_dir),
        "--output",
        str(output),
    ]
    if checkpoint:
        command.append("--checkpoint")
    completed = subprocess.run(command, capture_output=True, text=True, timeout=3600, check=False)
    if output.is_file():
        result = json.loads(output.read_text(encoding="utf-8"))
    else:
        result = {
            "schema_version": "darkmind-v2-phase3a-benchmark-profile-v1",
            "candidate": candidate,
            "micro_batch_size": batch_size,
            "gradient_checkpointing": checkpointing,
            "attention_implementation": attention,
            "hard_failure": "worker did not produce a result",
            "oom": False,
        }
    result["worker_exit_code"] = completed.returncode
    result["worker_stderr"] = completed.stderr.strip() or None
    return result


def validate_benchmark_report(payload: dict[str, Any], *, require_complete: bool = True) -> None:
    if payload.get("schema_version") != "darkmind-v2-phase3a-rtx4060-benchmark-v1":
        raise ValueError("unsupported benchmark schema")
    if payload.get("protocol", {}).get("warmup_microsteps", 0) < 10:
        raise ValueError("benchmark requires at least 10 warmup microsteps")
    if payload.get("protocol", {}).get("measured_microsteps", 0) < 30:
        raise ValueError("benchmark requires at least 30 measured microsteps")
    profiles = payload.get("profiles")
    if not isinstance(profiles, list):
        raise ValueError("benchmark profiles must be a list")
    if require_complete and {profile.get("candidate") for profile in profiles} != set(CANDIDATE_PATHS):
        raise ValueError("benchmark must include all candidates")
    for profile in profiles:
        if profile.get("micro_batch_size") not in {1, 2}:
            raise ValueError("unsafe benchmark batch size")
        if profile.get("attention_implementation") not in ATTENTION_PATHS:
            raise ValueError("unknown attention implementation")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 3A RTX 4060 Laptop Benchmark",
        "",
        f"GPU: {payload['environment']['gpu_name']}",
        "",
        "Each profile ran in an isolated process with deterministic synthetic token IDs, 10 warmup microsteps, and 30 measured optimizer microsteps. Checkpoint operations are excluded from active-step throughput.",
        "",
        payload["environment"]["determinism_note"],
        "",
        payload["environment"]["sdpa_note"],
        "",
        "| Candidate | MB | GC | Attention | tok/s | avg ms | p95 ms | Peak alloc | Peak reserved | Headroom | Result |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for profile in payload["profiles"]:
        failure = profile.get("hard_failure")
        lines.append(
            f"| {profile['candidate']} | {profile['micro_batch_size']} | "
            f"{'on' if profile['gradient_checkpointing'] else 'off'} | {profile['attention_implementation']} | "
            f"{profile.get('tokens_per_second', 0):,.1f} | "
            f"{profile.get('average_microstep_seconds', 0) * 1000:,.1f} | "
            f"{profile.get('p95_microstep_seconds', 0) * 1000:,.1f} | "
            f"{profile.get('peak_allocated_bytes', 0) / 2**30:.2f} GiB | "
            f"{profile.get('peak_reserved_bytes', 0) / 2**30:.2f} GiB | "
            f"{profile.get('vram_headroom_percent', 0):.1f}% | {failure or 'PASS'} |"
        )
    lines.extend(["", "## Candidate Checkpoints", ""])
    for candidate, checkpoint in payload["candidate_checkpoints"].items():
        if checkpoint is None:
            lines.append(f"- Candidate {candidate}: no successful checkpoint profile")
        else:
            lines.append(
                f"- Candidate {candidate}: {checkpoint['checkpoint_file_bytes'] / 2**20:.2f} MiB, "
                f"save {checkpoint['checkpoint_save_seconds']:.3f}s, reload {checkpoint['checkpoint_reload_seconds']:.3f}s"
            )
    lines.extend(
        [
            "",
            "## Protocol Notes",
            "",
            "Peak reserved VRAM is the safety basis because it captures allocator reservation. Micro-batch 2 was attempted only for matching micro-batch-1 profiles with at least 25% reserved-memory headroom. OOM and CUDA failures remain visible in the table and do not remove a candidate silently.",
            "",
        ]
    )
    return "\n".join(lines)


def run_orchestrator(args: argparse.Namespace) -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNTIME_ROOT / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    profiles: list[dict[str, Any]] = []
    checkpoints: dict[str, dict[str, Any] | None] = {candidate: None for candidate in CANDIDATE_PATHS}

    for candidate in CANDIDATE_PATHS:
        batch_one: list[dict[str, Any]] = []
        checkpoint_needed = True
        for checkpointing in CHECKPOINTING_VALUES:
            for attention in ATTENTION_PATHS:
                profile = _run_profile(
                    run_dir=run_dir,
                    candidate=candidate,
                    batch_size=1,
                    checkpointing=checkpointing,
                    attention=attention,
                    warmup=args.warmup,
                    steps=args.steps,
                    seed=args.seed,
                    checkpoint=checkpoint_needed,
                )
                profiles.append(profile)
                batch_one.append(profile)
                if profile.get("checkpoint_reload_success"):
                    checkpoints[candidate] = profile
                    checkpoint_needed = False
        for parent in batch_one:
            if (
                parent.get("hard_failure") is None
                and parent.get("vram_headroom_percent", 0) >= MIN_BATCH_TWO_HEADROOM_PERCENT
            ):
                profiles.append(
                    _run_profile(
                        run_dir=run_dir,
                        candidate=candidate,
                        batch_size=2,
                        checkpointing=bool(parent["gradient_checkpointing"]),
                        attention=str(parent["attention_implementation"]),
                        warmup=args.warmup,
                        steps=args.steps,
                        seed=args.seed,
                        checkpoint=False,
                    )
                )

    first_environment = next((profile for profile in profiles if profile.get("gpu_name")), {})
    payload = {
        "schema_version": "darkmind-v2-phase3a-rtx4060-benchmark-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "result": "PASS" if any(profile.get("hard_failure") is None for profile in profiles) else "FAIL",
        "environment": {
            "gpu_name": first_environment.get("gpu_name", "unknown"),
            "gpu_total_memory_bytes": first_environment.get("gpu_total_memory_bytes"),
            "torch_version": first_environment.get("torch_version"),
            "cuda_runtime_version": first_environment.get("cuda_runtime_version"),
        },
        "protocol": {
            "precision": "bf16",
            "sequence_length": 512,
            "warmup_microsteps": args.warmup,
            "measured_microsteps": args.steps,
            "optimizer": "AdamW",
            "gradient_clip_norm": 1.0,
            "seed": args.seed,
            "minimum_safe_vram_headroom_percent": MIN_SAFE_HEADROOM_PERCENT,
            "batch_two_trigger_headroom_percent": MIN_BATCH_TWO_HEADROOM_PERCENT,
            "active_throughput_excludes_checkpoint_time": True,
            "worker_isolation": True,
        },
        "runtime_directory": run_dir.relative_to(ROOT).as_posix(),
        "profiles": profiles,
        "candidate_checkpoints": {
            candidate: (
                None
                if profile is None
                else {
                    key: profile[key]
                    for key in (
                        "checkpoint_file_bytes",
                        "checkpoint_path",
                        "checkpoint_reload_seconds",
                        "checkpoint_reload_success",
                        "checkpoint_save_seconds",
                        "checkpoint_save_success",
                    )
                }
            )
            for candidate, profile in checkpoints.items()
        },
    }
    sanitize_benchmark_report(payload)
    validate_benchmark_report(payload)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_json(REPORT_DIR / "phase3a_rtx4060_benchmark.json", payload)
    (REPORT_DIR / "phase3a_rtx4060_benchmark.md").write_text(render_markdown(payload), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--candidate", choices=sorted(CANDIDATE_PATHS))
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--gradient-checkpointing", choices=("on", "off"), default="on")
    parser.add_argument("--attention", choices=ATTENTION_PATHS, default="sdpa")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sanitize-existing-report", action="store_true")
    args = parser.parse_args()
    if args.warmup < 10 or args.steps < 30:
        parser.error("Phase 3A requires at least 10 warmup and 30 measured microsteps")
    if args.worker and (args.candidate is None or args.run_dir is None or args.output is None):
        parser.error("worker mode requires candidate, run-dir, and output")
    return args


def main() -> None:
    args = parse_args()
    if args.sanitize_existing_report:
        path = REPORT_DIR / "phase3a_rtx4060_benchmark.json"
        payload = sanitize_benchmark_report(json.loads(path.read_text(encoding="utf-8")))
        validate_benchmark_report(payload)
        _atomic_json(path, payload)
        (REPORT_DIR / "phase3a_rtx4060_benchmark.md").write_text(render_markdown(payload), encoding="utf-8")
        print(json.dumps({"result": payload["result"], "sanitized": True}, indent=2))
        return
    if args.worker:
        result = run_worker(args)
        _atomic_json(args.output, result)
        print(json.dumps({"candidate": args.candidate, "result": result.get("hard_failure") or "PASS"}))
        return
    payload = run_orchestrator(args)
    print(json.dumps({"result": payload["result"], "profiles": len(payload["profiles"])}, indent=2))


if __name__ == "__main__":
    main()
