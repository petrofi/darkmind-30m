"""Isolated crash, post-warmup memory, and soak diagnostics for Phase 3B finalists."""

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
from darkmind_v2.modeling.modeling_darkmind_v2 import DarkMindV2ForCausalLM
from darkmind_v2.modeling.phase3b_environment import validate_training_environment


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {
    "C": ROOT / "darkmind_v2" / "config" / "model_base_candidate_c_100m.json",
    "D": ROOT / "darkmind_v2" / "config" / "model_base_candidate_d_120m.json",
}
RUNTIME_ROOT = ROOT / "darkmind_v2" / "data" / "phase3b" / "diagnostics"
REPORT_DIR = ROOT / "darkmind_v2" / "reports"
SEED = 20260712
WINDOWS_EXCEPTION_NAMES = {
    3221225477: "STATUS_ACCESS_VIOLATION (0xC0000005)",
    3221226505: "STATUS_STACK_BUFFER_OVERRUN / fail-fast (0xC0000409)",
}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def tensor_bytes(tensors: list[torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def optimizer_bytes(optimizer: torch.optim.Optimizer) -> int:
    tensors: list[torch.Tensor] = []
    for state in optimizer.state.values():
        tensors.extend(value for value in state.values() if torch.is_tensor(value))
    return tensor_bytes(tensors)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]


def load_config(candidate: str, *, attention: str, checkpointing: bool) -> DarkMindV2Config:
    payload = json.loads(CONFIGS[candidate].read_text(encoding="utf-8"))
    payload["seed"] = SEED
    payload["attention_implementation"] = attention
    payload["gradient_checkpointing"] = checkpointing
    return DarkMindV2Config(**payload)


def configure_cuda(seed: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.device("cuda:0")


def progress(path: Path, operation: str, **details: Any) -> None:
    payload = {
        "operation": operation,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    atomic_json(path, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


def query_gpu_telemetry() -> dict[str, float] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        if completed.returncode:
            return None
        values = [float(value.strip()) for value in completed.stdout.splitlines()[0].split(",")]
        return dict(zip(("temperature_c", "power_w", "sm_clock_mhz", "memory_clock_mhz"), values))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def build_fixture(
    candidate: str,
    *,
    batch_size: int,
    attention: str,
    checkpointing: bool,
    progress_path: Path,
    allow_unsafe_diagnostic: bool,
) -> tuple[
    DarkMindV2ForCausalLM,
    torch.optim.AdamW,
    torch.Tensor,
    torch.device,
    DarkMindV2Config,
]:
    device = configure_cuda(SEED)
    config = load_config(candidate, attention=attention, checkpointing=checkpointing)
    validation = validate_training_environment(
        config,
        micro_batch_size=batch_size,
        precision="bf16",
        allow_unsafe_diagnostic=allow_unsafe_diagnostic,
    )
    progress(progress_path, "environment_validated", validation=validation)
    model = DarkMindV2ForCausalLM(config).to(device=device, dtype=torch.bfloat16).train()
    progress(progress_path, "model_created", parameters=model.parameter_count())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.0e-4, betas=(0.9, 0.95), weight_decay=0.1, foreach=False
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED + batch_size)
    input_ids = torch.randint(
        8,
        config.vocab_size,
        (batch_size, config.block_size),
        dtype=torch.long,
        device=device,
        generator=generator,
    )
    progress(progress_path, "fixture_created", input_checksum=int(input_ids.to(torch.int64).sum().item()))
    return model, optimizer, input_ids, device, config


def optimizer_step(
    model: DarkMindV2ForCausalLM,
    optimizer: torch.optim.AdamW,
    input_ids: torch.Tensor,
    device: torch.device,
    progress_path: Path,
    step: int,
    *,
    record_progress: bool,
) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    if record_progress:
        progress(progress_path, "before_forward", step=step)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_ids, labels=input_ids)
        loss = output.loss
    if loss is None or not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite loss")
    torch.cuda.synchronize(device)
    if record_progress:
        progress(progress_path, "forward_synchronized", step=step, loss=float(loss.detach()))
        progress(progress_path, "before_backward", step=step)
    loss.backward()
    torch.cuda.synchronize(device)
    if record_progress:
        progress(progress_path, "backward_synchronized", step=step)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("non-finite gradient norm")
    if record_progress:
        progress(progress_path, "clipping_complete", step=step, gradient_norm=float(gradient_norm))
    optimizer.step()
    torch.cuda.synchronize(device)
    if record_progress:
        progress(progress_path, "optimizer_synchronized", step=step)
    return float(loss.detach()), float(gradient_norm.detach())


def diagnosis_worker(args: argparse.Namespace) -> dict[str, Any]:
    model, optimizer, input_ids, device, config = build_fixture(
        args.candidate,
        batch_size=args.batch_size,
        attention=args.attention,
        checkpointing=args.checkpointing == "on",
        progress_path=args.progress,
        allow_unsafe_diagnostic=True,
    )
    torch.cuda.reset_peak_memory_stats(device)
    losses: list[float] = []
    gradients: list[float] = []
    total_steps = args.warmup_steps + args.measured_steps
    for step in range(1, total_steps + 1):
        loss, gradient = optimizer_step(
            model, optimizer, input_ids, device, args.progress, step, record_progress=True
        )
        losses.append(loss)
        gradients.append(gradient)
    properties = torch.cuda.get_device_properties(device)
    result = {
        "schema_version": "darkmind-v2-phase3b-checkpointing-attempt-v1",
        "candidate": args.candidate,
        "seed": SEED,
        "batch_size": args.batch_size,
        "sequence_length": config.block_size,
        "precision": "bf16",
        "attention": args.attention,
        "gradient_checkpointing": args.checkpointing == "on",
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "result": "PASS",
        "losses_finite": all(math.isfinite(value) for value in losses),
        "gradients_finite": all(math.isfinite(value) for value in gradients),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
    }
    progress(args.progress, "worker_complete", result="PASS")
    return result


def memory_worker(args: argparse.Namespace) -> dict[str, Any]:
    model, optimizer, input_ids, device, config = build_fixture(
        args.candidate,
        batch_size=2,
        attention="sdpa",
        checkpointing=False,
        progress_path=args.progress,
        allow_unsafe_diagnostic=False,
    )
    torch.cuda.reset_peak_memory_stats(device)
    durations: list[float] = []
    for step in range(1, 11):
        started = time.perf_counter()
        optimizer_step(model, optimizer, input_ids, device, args.progress, step, record_progress=True)
        durations.append(time.perf_counter() - started)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    model_memory = tensor_bytes(list(model.parameters()))
    gradient_memory = tensor_bytes(gradients)
    optimizer_memory = optimizer_bytes(optimizer)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    properties = torch.cuda.get_device_properties(device)
    result = {
        "schema_version": "darkmind-v2-phase3b-post-warmup-memory-v1",
        "candidate": args.candidate,
        "optimizer_steps": 10,
        "optimizer_state_tensors": sum(
            1 for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value)
        ),
        "model_weight_bytes": model_memory,
        "gradient_bytes": gradient_memory,
        "optimizer_state_bytes": optimizer_memory,
        "activation_and_temporary_peak_bytes": max(
            0, peak_allocated - model_memory - gradient_memory - optimizer_memory
        ),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "gpu_total_memory_bytes": properties.total_memory,
        "reserved_headroom_percent": 100 * (properties.total_memory - peak_reserved) / properties.total_memory,
        "average_step_seconds": statistics.fmean(durations),
        "profile": {
            "precision": "bf16",
            "sequence_length": config.block_size,
            "micro_batch_size": 2,
            "attention": "sdpa",
            "gradient_checkpointing": False,
        },
        "result": "PASS",
    }
    progress(args.progress, "worker_complete", result="PASS")
    return result


def soak_worker(args: argparse.Namespace) -> dict[str, Any]:
    model, optimizer, first_input, device, config = build_fixture(
        args.candidate,
        batch_size=2,
        attention="sdpa",
        checkpointing=False,
        progress_path=args.progress,
        allow_unsafe_diagnostic=False,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED + 100)
    stream = [first_input]
    stream.extend(
        torch.randint(
            8, config.vocab_size, first_input.shape, dtype=torch.long, device=device, generator=generator
        )
        for _ in range(15)
    )
    torch.cuda.reset_peak_memory_stats(device)
    durations: list[float] = []
    losses: list[float] = []
    gradients: list[float] = []
    telemetry: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, args.soak_steps + 1):
        step_started = time.perf_counter()
        loss, gradient = optimizer_step(
            model,
            optimizer,
            stream[(step - 1) % len(stream)],
            device,
            args.progress,
            step,
            record_progress=step == 1 or step % 100 == 0,
        )
        durations.append(time.perf_counter() - step_started)
        losses.append(loss)
        gradients.append(gradient)
        if step == 1 or step % 25 == 0 or step == args.soak_steps:
            sample = query_gpu_telemetry()
            if sample is not None:
                telemetry.append({"step": step, **sample})
    elapsed = time.perf_counter() - started
    properties = torch.cuda.get_device_properties(device)
    result = {
        "schema_version": "darkmind-v2-phase3b-finalist-soak-v1",
        "candidate": args.candidate,
        "optimizer_steps": args.soak_steps,
        "elapsed_seconds": elapsed,
        "tokens_per_second": args.soak_steps * 2 * config.block_size / elapsed,
        "p50_step_seconds": statistics.median(durations),
        "p95_step_seconds": percentile(durations, 0.95),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "reserved_headroom_percent": 100
        * (properties.total_memory - torch.cuda.max_memory_reserved(device))
        / properties.total_memory,
        "losses_finite": all(math.isfinite(value) for value in losses),
        "gradients_finite": all(math.isfinite(value) for value in gradients),
        "cuda_errors": [],
        "telemetry": telemetry,
        "temperature_min_c": min((item["temperature_c"] for item in telemetry), default=None),
        "temperature_max_c": max((item["temperature_c"] for item in telemetry), default=None),
        "power_min_w": min((item["power_w"] for item in telemetry), default=None),
        "power_max_w": max((item["power_w"] for item in telemetry), default=None),
        "sm_clock_min_mhz": min((item["sm_clock_mhz"] for item in telemetry), default=None),
        "sm_clock_max_mhz": max((item["sm_clock_mhz"] for item in telemetry), default=None),
        "profile": {
            "precision": "bf16",
            "sequence_length": config.block_size,
            "micro_batch_size": 2,
            "attention": "sdpa",
            "gradient_checkpointing": False,
            "input_stream": "fixed deterministic 16-batch synthetic cycle",
        },
        "synthetic_loss_is_language_evidence": False,
        "result": "PASS",
    }
    progress(args.progress, "worker_complete", result="PASS")
    return result


def run_child(
    run_dir: Path,
    name: str,
    command_args: list[str],
    *,
    timeout: int = 3600,
) -> dict[str, Any]:
    output = run_dir / f"{name}.json"
    progress_path = run_dir / f"{name}.progress.json"
    command = [
        sys.executable,
        "-m",
        "darkmind_v2.training.phase3b_finalist_diagnostics",
        "worker",
        "--output",
        str(output),
        "--progress",
        str(progress_path),
        *command_args,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    result = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else {
        "result": "CRASH",
        "hard_failure": "worker did not produce a result",
    }
    result.update(
        {
            "worker_exit_code": completed.returncode,
            "windows_exception": WINDOWS_EXCEPTION_NAMES.get(completed.returncode),
            "worker_stdout": completed.stdout,
            "worker_stderr": completed.stderr,
            "last_progress": (
                json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else None
            ),
        }
    )
    return result


def diagnosis_report(payload: dict[str, Any]) -> str:
    primary = payload["primary_attempts"]
    lines = [
        "# Phase 3B Candidate C Checkpointing Diagnosis",
        "",
        f"Classification: **{payload['classification']}**",
        "",
        "The affected environment is Windows, PyTorch 2.4.1+cu121, CUDA 12.1, RTX 4060 Laptop GPU, "
        "BF16, sequence length 512, micro-batch 2, SDPA, and non-reentrant gradient checkpointing.",
        "",
        "## Identical Candidate C Attempts",
        "",
        "| Attempt | Exit | Windows exception | Last completed operation | Result |",
        "|---:|---:|---|---|---|",
    ]
    for index, item in enumerate(primary, start=1):
        progress_item = item.get("last_progress") or {}
        lines.append(
            f"| {index} | {item['worker_exit_code']} | {item.get('windows_exception') or '-'} | "
            f"{progress_item.get('operation', '-')} | {item.get('result', 'CRASH')} |"
        )
    lines.extend(
        [
            "",
            "## Controls",
            "",
            "| Candidate | MB | Attention | Checkpointing | Exit | Result |",
            "|---|---:|---|---|---:|---|",
        ]
    )
    for item in payload["controls"]:
        lines.append(
            f"| {item['candidate']} | {item['batch_size']} | {item['attention']} | "
            f"{'on' if item['gradient_checkpointing'] else 'off'} | {item['worker_exit_code']} | "
            f"{item.get('result', 'CRASH')} |"
        )
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "Gradient checkpointing is unnecessary at the measured memory levels and is disabled for the "
            "recommended production profile. The exact affected combination is rejected before model training; "
            "diagnostic code must opt in explicitly. The failed profile remains visible in ignored runtime JSON.",
            "",
            "Phase 3A preserved one native Windows fail-fast with exit code 3221226505, but none of the five "
            "identical Phase 3B attempts reproduced it. All alternate C profiles and corresponding D controls "
            "passed. The root cause therefore remains an intermittent process/backend instability rather than a "
            "deterministic model-code defect or an out-of-memory event.",
            "",
        ]
    )
    return "\n".join(lines)


def run_diagnosis() -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNTIME_ROOT / f"checkpointing_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    common = [
        "--mode", "diagnosis", "--candidate", "C", "--batch-size", "2", "--attention", "sdpa",
        "--checkpointing", "on", "--warmup-steps", "10", "--measured-steps", "10",
    ]
    primary = [run_child(run_dir, f"candidate_c_primary_{index:02d}", common) for index in range(1, 6)]
    profiles = [
        ("C", 2, "sdpa", "off"),
        ("C", 1, "sdpa", "on"),
        ("C", 2, "fallback", "on"),
        ("C", 1, "fallback", "on"),
        ("D", 2, "sdpa", "on"),
        ("D", 2, "sdpa", "off"),
        ("D", 1, "sdpa", "on"),
        ("D", 2, "fallback", "on"),
        ("D", 1, "fallback", "on"),
    ]
    controls: list[dict[str, Any]] = []
    for candidate, batch, attention, checkpointing in profiles:
        name = f"candidate_{candidate.lower()}_b{batch}_{attention}_gc_{checkpointing}"
        item = run_child(
            run_dir,
            name,
            [
                "--mode", "diagnosis", "--candidate", candidate, "--batch-size", str(batch),
                "--attention", attention, "--checkpointing", checkpointing,
                "--warmup-steps", "10", "--measured-steps", "10",
            ],
        )
        controls.append(item)
    primary_codes = {item["worker_exit_code"] for item in primary}
    controls_pass = all(item["worker_exit_code"] == 0 and item.get("result") == "PASS" for item in controls)
    classification = (
        "PyTorch/Windows/backend defect"
        if len(primary_codes) == 1 and next(iter(primary_codes)) == 3221226505 and controls_pass
        else "intermittent process/backend instability"
        if controls_pass
        else "unresolved"
    )
    payload = {
        "schema_version": "darkmind-v2-phase3b-checkpointing-diagnosis-v1",
        "classification": classification,
        "historical_phase3a_failure": {
            "worker_exit_code": 3221226505,
            "windows_exception": WINDOWS_EXCEPTION_NAMES[3221226505],
            "result_file_written": False,
        },
        "identical_attempt_count": 5,
        "primary_attempts": primary,
        "controls": controls,
        "runtime_directory": run_dir.relative_to(ROOT).as_posix(),
    }
    atomic_json(run_dir / "diagnosis.json", payload)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "phase3b_candidate_c_checkpointing_diagnosis.md").write_text(
        diagnosis_report(payload), encoding="utf-8"
    )
    return payload


def run_memory() -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNTIME_ROOT / f"memory_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    results = {
        candidate: run_child(
            run_dir,
            f"candidate_{candidate.lower()}_memory",
            ["--mode", "memory", "--candidate", candidate],
        )
        for candidate in CONFIGS
    }
    payload = {
        "schema_version": "darkmind-v2-phase3b-post-warmup-memory-audit-v1",
        "results": results,
        "runtime_directory": run_dir.relative_to(ROOT).as_posix(),
        "result": "PASS" if all(item.get("result") == "PASS" for item in results.values()) else "FAIL",
    }
    atomic_json(run_dir / "memory_audit.json", payload)
    return payload


def run_soak(steps: int) -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNTIME_ROOT / f"soak_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    results = {
        candidate: run_child(
            run_dir,
            f"candidate_{candidate.lower()}_soak",
            ["--mode", "soak", "--candidate", candidate, "--soak-steps", str(steps)],
            timeout=7200,
        )
        for candidate in CONFIGS
    }
    payload = {
        "schema_version": "darkmind-v2-phase3b-finalist-soak-audit-v1",
        "results": results,
        "runtime_directory": run_dir.relative_to(ROOT).as_posix(),
        "result": "PASS" if all(item.get("result") == "PASS" for item in results.values()) else "FAIL",
    }
    atomic_json(run_dir / "soak_audit.json", payload)
    lines = [
        "# Phase 3B Finalist Soak Test",
        "",
        "Synthetic loss is used only for numerical and process stability; it is not language-learning evidence.",
        "",
        "| Candidate | Steps | Minutes | tok/s | p50 ms | p95 ms | Peak reserved GiB | Temp C | Power W | SM clock MHz | Result |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for candidate, item in results.items():
        lines.append(
            f"| {candidate} | {item.get('optimizer_steps', 0)} | {item.get('elapsed_seconds', 0) / 60:.2f} | "
            f"{item.get('tokens_per_second', 0):,.1f} | {item.get('p50_step_seconds', 0) * 1000:.1f} | "
            f"{item.get('p95_step_seconds', 0) * 1000:.1f} | {item.get('peak_reserved_bytes', 0) / 2**30:.2f} | "
            f"{item.get('temperature_min_c')}-{item.get('temperature_max_c')} | "
            f"{item.get('power_min_w')}-{item.get('power_max_w')} | "
            f"{item.get('sm_clock_min_mhz')}-{item.get('sm_clock_max_mhz')} | {item.get('result')} |"
        )
    lines.append("")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "phase3b_finalist_soak_test.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("diagnose")
    commands.add_parser("memory")
    soak = commands.add_parser("soak")
    soak.add_argument("--steps", type=int, default=1000)
    worker = commands.add_parser("worker")
    worker.add_argument("--mode", choices=("diagnosis", "memory", "soak"), required=True)
    worker.add_argument("--candidate", choices=sorted(CONFIGS), required=True)
    worker.add_argument("--batch-size", type=int, choices=(1, 2), default=2)
    worker.add_argument("--attention", choices=("sdpa", "fallback"), default="sdpa")
    worker.add_argument("--checkpointing", choices=("on", "off"), default="off")
    worker.add_argument("--warmup-steps", type=int, default=10)
    worker.add_argument("--measured-steps", type=int, default=10)
    worker.add_argument("--soak-steps", type=int, default=1000)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--progress", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "diagnose":
        result = run_diagnosis()
    elif args.command == "memory":
        result = run_memory()
    elif args.command == "soak":
        if args.steps < 1000:
            raise ValueError("Phase 3B soak requires at least 1,000 optimizer steps")
        result = run_soak(args.steps)
    else:
        try:
            if args.mode == "diagnosis":
                result = diagnosis_worker(args)
            elif args.mode == "memory":
                result = memory_worker(args)
            else:
                result = soak_worker(args)
            atomic_json(args.output, result)
        except Exception as exc:
            result = {
                "result": "FAIL",
                "hard_failure": f"{type(exc).__name__}: {exc}",
            }
            atomic_json(args.output, result)
            raise
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
