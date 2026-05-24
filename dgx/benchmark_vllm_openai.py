#!/usr/bin/env python3
import argparse
import concurrent.futures
import csv
import datetime as dt
import json
import os
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class Result:
    request_id: int
    prompt_id: int
    ok: bool
    latency_s: float
    ttft_s: float | None
    output_tokens: int
    prompt_tokens: int | None
    total_tokens: int | None
    error: str | None = None


@dataclass
class ProcTimes:
    total: int
    idle: int


@dataclass
class ProcessSnapshot:
    cpu_ticks: int
    rss_bytes: int


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    low = int(k)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (k - low)


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def parse_sse_line(line: bytes) -> dict[str, Any] | None:
    if not line.startswith(b"data: "):
        return None
    payload = line[len(b"data: ") :].strip()
    if payload == b"[DONE]":
        return None
    return json.loads(payload)


def read_proc_stat() -> ProcTimes | None:
    try:
        parts = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
    except OSError:
        return None
    values = [int(part) for part in parts]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return ProcTimes(total=sum(values), idle=idle)


def read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        key, raw_value = line.split(":", 1)
        fields = raw_value.strip().split()
        if fields:
            values[key] = int(fields[0]) * 1024
    return values


def system_cpu_percent(prev: ProcTimes | None, cur: ProcTimes | None) -> float | None:
    if prev is None or cur is None:
        return None
    total_delta = cur.total - prev.total
    idle_delta = cur.idle - prev.idle
    if total_delta <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))


def read_gpu_samples() -> tuple[list[dict[str, Any]], str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [], str(exc)

    samples = []
    def parse_float(raw: str) -> float | None:
        if raw in {"[N/A]", "N/A", "Not Supported", ""}:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        util_pct = parse_float(parts[1])
        mem_used_mib = parse_float(parts[2])
        mem_total_mib = parse_float(parts[3])
        mem_used_bytes = mem_used_mib * 1024 * 1024 if mem_used_mib is not None else None
        mem_total_bytes = mem_total_mib * 1024 * 1024 if mem_total_mib is not None else None
        samples.append(
            {
                "index": index,
                "utilization_gpu_pct": util_pct,
                "memory_used_bytes": mem_used_bytes,
                "memory_total_bytes": mem_total_bytes,
                "memory_used_pct": (
                    100.0 * mem_used_bytes / mem_total_bytes
                    if mem_used_bytes is not None and mem_total_bytes
                    else None
                ),
            }
        )
    return samples, None


def read_gpu_process_memory(pids: list[int]) -> tuple[int | None, str | None]:
    if not pids:
        return 0, None
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)

    pid_set = set(pids)
    total_mib = 0
    found = False
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            used_mib = int(parts[-1])
        except ValueError:
            continue
        if pid in pid_set:
            total_mib += used_mib
            found = True
    return (total_mib * 1024 * 1024 if found else 0), None


def find_pids(match: str | None, explicit_pids: list[int]) -> list[int]:
    pids = set(explicit_pids)
    if match:
        needle = match.lower()
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(
                    "utf-8", errors="ignore"
                )
            except OSError:
                continue
            haystack = cmdline.lower()
            if needle in haystack and "benchmark_vllm_openai.py" not in haystack:
                pids.add(int(entry.name))
    return sorted(pids)


def read_process_snapshot(pids: list[int]) -> ProcessSnapshot:
    ticks = 0
    rss = 0
    page_size = os.sysconf("SC_PAGE_SIZE")
    for pid in pids:
        proc = Path("/proc") / str(pid)
        try:
            stat = (proc / "stat").read_text(encoding="utf-8")
            fields = stat.rsplit(")", 1)[1].split()
            # fields are stat columns from state onward; utime is col 14, stime is col 15.
            ticks += int(fields[11]) + int(fields[12])
            statm = (proc / "statm").read_text(encoding="utf-8").split()
            rss += int(statm[1]) * page_size
        except (OSError, IndexError, ValueError):
            continue
    return ProcessSnapshot(cpu_ticks=ticks, rss_bytes=rss)


class ResourceMonitor:
    def __init__(
        self,
        interval_s: float,
        process_match: str | None,
        pids: list[int],
        max_host_ram_pct: float | None = None,
        max_swap_used_gib: float | None = None,
        max_swap_growth_gib: float | None = None,
        guard_grace_samples: int = 2,
    ):
        self.interval_s = interval_s
        self.process_match = process_match
        self.explicit_pids = pids
        self.max_host_ram_pct = max_host_ram_pct
        self.max_swap_used_bytes = (
            int(max_swap_used_gib * 1024**3) if max_swap_used_gib is not None else None
        )
        self.max_swap_growth_bytes = (
            int(max_swap_growth_gib * 1024**3) if max_swap_growth_gib is not None else None
        )
        self.guard_grace_samples = max(1, guard_grace_samples)
        self.samples: list[dict[str, Any]] = []
        self.abort_reason: str | None = None
        self._guard_hits = 0
        self._initial_swap_used_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(2.0, self.interval_s * 2))
        return self.summary()

    def _run(self) -> None:
        prev_system = read_proc_stat()
        prev_process: ProcessSnapshot | None = None
        prev_t = time.perf_counter()
        clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        cpu_count = os.cpu_count() or 1

        while not self._stop.wait(self.interval_s):
            now = time.perf_counter()
            mem = read_meminfo()
            cur_system = read_proc_stat()
            pids = find_pids(self.process_match, self.explicit_pids)
            gpu_samples, gpu_error = read_gpu_samples()
            process_gpu_memory, process_gpu_error = read_gpu_process_memory(pids)
            cur_process = read_process_snapshot(pids)

            process_cpu = None
            if prev_process is not None:
                elapsed = max(now - prev_t, 0.000001)
                tick_delta = cur_process.cpu_ticks - prev_process.cpu_ticks
                process_cpu = 100.0 * tick_delta / (clock_ticks * elapsed * cpu_count)

            total_mem = mem.get("MemTotal")
            available_mem = mem.get("MemAvailable")
            swap_total = mem.get("SwapTotal")
            swap_free = mem.get("SwapFree")
            used_mem = None
            used_mem_pct = None
            if total_mem is not None and available_mem is not None:
                used_mem = total_mem - available_mem
                used_mem_pct = 100.0 * used_mem / total_mem
            swap_used = None
            swap_used_pct = None
            if swap_total is not None and swap_free is not None:
                swap_used = swap_total - swap_free
                swap_used_pct = 100.0 * swap_used / swap_total if swap_total else 0.0
                if self._initial_swap_used_bytes is None:
                    self._initial_swap_used_bytes = swap_used

            self.samples.append(
                {
                    "t_s": now,
                    "system_cpu_pct": system_cpu_percent(prev_system, cur_system),
                    "system_mem_used_bytes": used_mem,
                    "system_mem_used_pct": used_mem_pct,
                    "system_swap_used_bytes": swap_used,
                    "system_swap_used_pct": swap_used_pct,
                    "process_match": self.process_match,
                    "process_pids": pids,
                    "process_count": len(pids),
                    "process_cpu_pct_of_host": process_cpu,
                    "process_rss_bytes": cur_process.rss_bytes,
                    "gpus": gpu_samples,
                    "gpu_error": gpu_error,
                    "process_gpu_memory_bytes": process_gpu_memory,
                    "process_gpu_error": process_gpu_error,
                }
            )
            self._check_guard(used_mem_pct, swap_used)
            prev_system = cur_system
            prev_process = cur_process
            prev_t = now

    def _check_guard(self, used_mem_pct: float | None, swap_used: int | None) -> None:
        if self.abort_reason:
            return
        reasons = []
        if (
            self.max_host_ram_pct is not None
            and used_mem_pct is not None
            and used_mem_pct >= self.max_host_ram_pct
        ):
            reasons.append(
                f"host RAM {used_mem_pct:.2f}% >= guard {self.max_host_ram_pct:.2f}%"
            )
        if (
            self.max_swap_used_bytes is not None
            and swap_used is not None
            and swap_used >= self.max_swap_used_bytes
        ):
            reasons.append(
                f"swap used {swap_used / 1024**3:.2f} GiB >= guard "
                f"{self.max_swap_used_bytes / 1024**3:.2f} GiB"
            )
        if (
            self.max_swap_growth_bytes is not None
            and swap_used is not None
            and self._initial_swap_used_bytes is not None
            and swap_used - self._initial_swap_used_bytes >= self.max_swap_growth_bytes
        ):
            reasons.append(
                f"swap grew {(swap_used - self._initial_swap_used_bytes) / 1024**3:.2f} GiB "
                f">= guard {self.max_swap_growth_bytes / 1024**3:.2f} GiB"
            )

        if reasons:
            self._guard_hits += 1
            if self._guard_hits >= self.guard_grace_samples:
                self.abort_reason = "; ".join(reasons)
                self._stop.set()
        else:
            self._guard_hits = 0

    def summary(self) -> dict[str, Any]:
        def values(key: str) -> list[float]:
            return [sample[key] for sample in self.samples if sample.get(key) is not None]

        system_cpu = values("system_cpu_pct")
        system_mem_pct = values("system_mem_used_pct")
        system_mem = values("system_mem_used_bytes")
        system_swap_pct = values("system_swap_used_pct")
        system_swap = values("system_swap_used_bytes")
        proc_cpu = values("process_cpu_pct_of_host")
        proc_rss = values("process_rss_bytes")
        gpu_util_mean_by_sample = []
        gpu_util_max_by_sample = []
        gpu_mem_used_by_sample = []
        gpu_mem_pct_max_by_sample = []
        process_gpu_memory = values("process_gpu_memory_bytes")
        gpu_errors = []
        for sample in self.samples:
            if sample.get("process_gpu_error"):
                gpu_errors.append(sample["process_gpu_error"])
            if sample.get("gpu_error"):
                gpu_errors.append(sample["gpu_error"])
            gpus = sample.get("gpus") or []
            if not gpus:
                continue
            utils = [gpu["utilization_gpu_pct"] for gpu in gpus if gpu.get("utilization_gpu_pct") is not None]
            mem_used = [gpu["memory_used_bytes"] for gpu in gpus if gpu.get("memory_used_bytes") is not None]
            mem_pct = [gpu["memory_used_pct"] for gpu in gpus if gpu.get("memory_used_pct") is not None]
            if utils:
                gpu_util_mean_by_sample.append(statistics.mean(utils))
                gpu_util_max_by_sample.append(max(utils))
            if mem_used:
                gpu_mem_used_by_sample.append(sum(mem_used))
            if mem_pct:
                gpu_mem_pct_max_by_sample.append(max(mem_pct))
        return {
            "sample_count": len(self.samples),
            "sample_interval_s": self.interval_s,
            "system_cpu_pct": {
                "mean": mean(system_cpu),
                "max": max(system_cpu) if system_cpu else None,
            },
            "system_mem_used_pct": {
                "mean": mean(system_mem_pct),
                "max": max(system_mem_pct) if system_mem_pct else None,
            },
            "system_mem_used_bytes": {
                "mean": mean(system_mem),
                "max": max(system_mem) if system_mem else None,
            },
            "system_swap_used_pct": {
                "mean": mean(system_swap_pct),
                "max": max(system_swap_pct) if system_swap_pct else None,
            },
            "system_swap_used_bytes": {
                "mean": mean(system_swap),
                "max": max(system_swap) if system_swap else None,
            },
            "process_cpu_pct_of_host": {
                "mean": mean(proc_cpu),
                "max": max(proc_cpu) if proc_cpu else None,
            },
            "process_rss_bytes": {
                "mean": mean(proc_rss),
                "max": max(proc_rss) if proc_rss else None,
            },
            "gpu_utilization_pct": {
                "mean_across_gpus_mean": mean(gpu_util_mean_by_sample),
                "max_gpu_mean": mean(gpu_util_max_by_sample),
                "max_gpu_max": max(gpu_util_max_by_sample) if gpu_util_max_by_sample else None,
            },
            "gpu_memory_used_bytes": {
                "mean_total": mean(gpu_mem_used_by_sample),
                "max_total": max(gpu_mem_used_by_sample) if gpu_mem_used_by_sample else None,
            },
            "gpu_memory_used_pct": {
                "max_gpu_mean": mean(gpu_mem_pct_max_by_sample),
                "max_gpu_max": max(gpu_mem_pct_max_by_sample) if gpu_mem_pct_max_by_sample else None,
            },
            "process_gpu_memory_bytes": {
                "mean": mean(process_gpu_memory),
                "max": max(process_gpu_memory) if process_gpu_memory else None,
            },
            "gpu_errors": sorted(set(gpu_errors))[:3],
            "last_process_pids": self.samples[-1].get("process_pids", []) if self.samples else [],
            "last_gpus": self.samples[-1].get("gpus", []) if self.samples else [],
            "abort_reason": self.abort_reason,
            "samples": self.samples,
        }


def run_one(
    request_id: int,
    prompt_id: int,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> Result:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    start = time.perf_counter()
    first_token_at = None
    output_tokens = 0
    prompt_tokens = None
    total_tokens = None

    try:
        with requests.post(url, json=body, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                event = parse_sse_line(line)
                if not event:
                    continue

                usage = event.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens")
                    output_tokens = usage.get("completion_tokens", output_tokens)
                    total_tokens = usage.get("total_tokens")
                    continue

                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta and first_token_at is None:
                    first_token_at = time.perf_counter()
                content = delta.get("content")
                if content:
                    output_tokens += 1
    except Exception as exc:
        return Result(
            request_id=request_id,
            prompt_id=prompt_id,
            ok=False,
            latency_s=time.perf_counter() - start,
            ttft_s=None,
            output_tokens=0,
            prompt_tokens=None,
            total_tokens=None,
            error=str(exc),
        )

    end = time.perf_counter()
    return Result(
        request_id=request_id,
        prompt_id=prompt_id,
        ok=True,
        latency_s=end - start,
        ttft_s=None if first_token_at is None else first_token_at - start,
        output_tokens=output_tokens,
        prompt_tokens=prompt_tokens,
        total_tokens=total_tokens,
    )


def per_request_stats(result: Result) -> dict[str, Any]:
    decode_s = None
    decode_tok_s = None
    if result.ttft_s is not None:
        decode_s = max(result.latency_s - result.ttft_s, 0.0)
        if decode_s > 0 and result.output_tokens > 1:
            decode_tok_s = (result.output_tokens - 1) / decode_s

    return {
        "request_id": result.request_id,
        "prompt_id": result.prompt_id,
        "ok": result.ok,
        "latency_s": result.latency_s,
        "ttft_s": result.ttft_s,
        "decode_s": decode_s,
        "output_tokens": result.output_tokens,
        "prompt_tokens": result.prompt_tokens,
        "total_tokens": result.total_tokens,
        "end_to_end_output_tok_s": (
            result.output_tokens / result.latency_s if result.latency_s else None
        ),
        "decode_output_tok_s": decode_tok_s,
        "error": result.error,
    }


def summarize(
    results: list[Result],
    wall_s: float,
    model: str,
    base_url: str,
    concurrency: int,
    prompt_words: int,
    max_tokens: int,
    prompt_mode: str,
    prompt_count: int,
    include_per_request: bool,
    resources: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    failures = [r for r in results if not r.ok]
    latencies = [r.latency_s for r in ok]
    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    e2e_tok_s = [r.output_tokens / r.latency_s for r in ok if r.latency_s > 0]
    prefill_tok_s = [
        r.prompt_tokens / r.ttft_s
        for r in ok
        if r.prompt_tokens and r.ttft_s is not None and r.ttft_s > 0
    ]
    decode_tok_s = [
        (r.output_tokens - 1) / (r.latency_s - r.ttft_s)
        for r in ok
        if r.ttft_s is not None
        and r.output_tokens > 1
        and r.latency_s > r.ttft_s
    ]
    output_tokens = sum(r.output_tokens for r in ok)
    prompt_tokens = sum(r.prompt_tokens or 0 for r in ok)
    total_tokens = sum(r.total_tokens or 0 for r in ok)

    summary: dict[str, Any] = {
        "model": model,
        "base_url": base_url,
        "concurrency": concurrency,
        "requests": len(results),
        "successful_requests": len(ok),
        "failed_requests": len(failures),
        "prompt_mode": prompt_mode,
        "prompt_count": prompt_count,
        "prompt_words": prompt_words,
        "max_tokens": max_tokens,
        "wall_time_s": wall_s,
        "benchmark_time_s": wall_s,
        "request_throughput_rps": len(ok) / wall_s if wall_s else None,
        "output_token_throughput_tok_s": output_tokens / wall_s if wall_s else None,
        "prompt_token_throughput_tok_s": prompt_tokens / wall_s if prompt_tokens and wall_s else None,
        "prefill_prompt_tok_s_approx": {
            "mean": mean(prefill_tok_s),
            "p50": percentile(prefill_tok_s, 50),
            "p90": percentile(prefill_tok_s, 90),
            "p99": percentile(prefill_tok_s, 99),
        },
        "total_token_throughput_tok_s": total_tokens / wall_s if total_tokens and wall_s else None,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency_s": {
            "mean": mean(latencies),
            "p50": percentile(latencies, 50),
            "p90": percentile(latencies, 90),
            "p99": percentile(latencies, 99),
        },
        "ttft_s": {
            "mean": mean(ttfts),
            "p50": percentile(ttfts, 50),
            "p90": percentile(ttfts, 90),
            "p99": percentile(ttfts, 99),
        },
        "per_request_end_to_end_output_tok_s": {
            "mean": mean(e2e_tok_s),
            "p50": percentile(e2e_tok_s, 50),
            "p90": percentile(e2e_tok_s, 90),
            "p99": percentile(e2e_tok_s, 99),
        },
        "per_request_decode_output_tok_s": {
            "mean": mean(decode_tok_s),
            "p50": percentile(decode_tok_s, 50),
            "p90": percentile(decode_tok_s, 90),
            "p99": percentile(decode_tok_s, 99),
        },
        "resources": resources,
        "metadata": metadata or {},
        "aborted": bool(resources and resources.get("abort_reason")),
        "abort_reason": resources.get("abort_reason") if resources else None,
        "errors": [r.error for r in failures[:5]],
    }

    if include_per_request:
        summary["per_request"] = [
            per_request_stats(result)
            for result in sorted(results, key=lambda item: item.request_id)
        ]

    return summary


def make_prompt(target_words: int, prompt_id: int, unique: bool) -> str:
    if unique:
        seed = (
            f"Benchmark sample {prompt_id:06d}. "
            f"Scenario code {prompt_id * 7919 % 104729}. "
            "Assess a production inference system that serves varied user traffic. "
            "Discuss latency, batching, queueing, memory pressure, scheduling, "
            "throughput, monitoring, and capacity planning with concrete wording. "
        )
    else:
        seed = (
            "Analyze inference serving performance for a large language model. "
            "Use concise technical language and include concrete observations. "
        )
    words = seed.split()
    repeated: list[str] = []
    while len(repeated) < target_words:
        repeated.extend(words)
    body = " ".join(repeated[:target_words])
    return (
        f"{body}\n\n"
        "Return a detailed technical analysis. Continue until the token budget is used."
    )


def make_prompts(target_words: int, count: int, mode: str) -> list[str]:
    if mode == "repeated":
        return [make_prompt(target_words, 0, unique=False)]
    return [make_prompt(target_words, prompt_id, unique=True) for prompt_id in range(count)]


def run_benchmark(args: argparse.Namespace, concurrency: int, requests_count: int) -> dict[str, Any]:
    prompt_count = args.prompt_count or requests_count
    prompts = make_prompts(args.prompt_words, prompt_count, args.prompt_mode)

    for warmup_id in range(args.warmup):
        prompt_id = warmup_id % len(prompts)
        warmup = run_one(
            -warmup_id - 1,
            prompt_id,
            args.base_url,
            args.model,
            prompts[prompt_id],
            min(args.max_tokens, 32),
            args.temperature,
            args.timeout,
        )
        if not warmup.ok:
            raise SystemExit(f"warmup failed: {warmup.error}")

    host_ram_guard_pct = args.max_host_ram_pct if args.max_host_ram_pct and args.max_host_ram_pct > 0 else None
    monitor = ResourceMonitor(
        args.resource_interval,
        args.process_match,
        args.pid,
        max_host_ram_pct=host_ram_guard_pct,
        max_swap_used_gib=args.max_swap_used_gib,
        max_swap_growth_gib=args.max_swap_growth_gib,
        guard_grace_samples=args.guard_grace_samples,
    )
    monitor.start()
    started = time.perf_counter()
    results: list[Result] = []
    submitted: set[int] = set()
    completed_ids: set[int] = set()
    next_request_id = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures: dict[concurrent.futures.Future[Result], int] = {}

        def submit_more() -> None:
            nonlocal next_request_id
            while (
                not monitor.abort_reason
                and next_request_id < requests_count
                and len(futures) < concurrency
            ):
                request_id = next_request_id
                next_request_id += 1
                submitted.add(request_id)
                futures[
                    pool.submit(
                        run_one,
                        request_id,
                        request_id % len(prompts),
                        args.base_url,
                        args.model,
                        prompts[request_id % len(prompts)],
                        args.max_tokens,
                        args.temperature,
                        args.timeout,
                    )
                ] = request_id

        submit_more()
        while futures:
            done, _ = concurrent.futures.wait(
                futures,
                timeout=max(0.2, min(1.0, args.resource_interval)),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if monitor.abort_reason:
                for future in list(futures):
                    if future.cancel():
                        request_id = futures.pop(future)
                        completed_ids.add(request_id)
                        results.append(
                            Result(
                                request_id=request_id,
                                prompt_id=request_id % len(prompts),
                                ok=False,
                                latency_s=0.0,
                                ttft_s=None,
                                output_tokens=0,
                                prompt_tokens=None,
                                total_tokens=None,
                                error=f"cancelled after resource guard tripped: {monitor.abort_reason}",
                            )
                        )
                # Let already-running requests finish, but do not submit more.

            for future in done:
                request_id = futures.pop(future)
                completed_ids.add(request_id)
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(
                        Result(
                            request_id=request_id,
                            prompt_id=request_id % len(prompts),
                            ok=False,
                            latency_s=0.0,
                            ttft_s=None,
                            output_tokens=0,
                            prompt_tokens=None,
                            total_tokens=None,
                            error=str(exc),
                        )
                    )
            submit_more()

    if monitor.abort_reason:
        for request_id in range(requests_count):
            if request_id not in submitted and request_id not in completed_ids:
                results.append(
                    Result(
                        request_id=request_id,
                        prompt_id=request_id % len(prompts),
                        ok=False,
                        latency_s=0.0,
                        ttft_s=None,
                        output_tokens=0,
                        prompt_tokens=None,
                        total_tokens=None,
                        error=f"not launched after resource guard tripped: {monitor.abort_reason}",
                    )
                )

    wall_s = time.perf_counter() - started
    resources = monitor.stop()
    metadata = {
        "vllm_max_concurrency": args.vllm_max_concurrency,
        "vllm_max_model_len": args.vllm_max_model_len,
        "concurrency_safety_fraction": args.concurrency_safety_fraction,
        "host_ram_guard_pct": host_ram_guard_pct,
        "swap_used_guard_gib": args.max_swap_used_gib,
        "swap_growth_guard_gib": args.max_swap_growth_gib,
    }

    return summarize(
        results,
        wall_s,
        args.model,
        args.base_url,
        concurrency,
        args.prompt_words,
        args.max_tokens,
        args.prompt_mode,
        len(prompts),
        not args.no_per_request_details,
        resources,
        metadata,
    )

def nested_get(data: dict[str, Any], path: str) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def bytes_to_gib(value: Any) -> float | None:
    if value is None:
        return None
    return float(value) / (1024**3)


def table_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in summaries:
        resources = item.get("resources") or {}
        rows.append(
            {
                "model": item.get("model"),
                "mode": item.get("prompt_mode"),
                "conc": item.get("concurrency"),
                "req": item.get("requests"),
                "fail": item.get("failed_requests"),
                "aborted": item.get("aborted"),
                "abort_reason": item.get("abort_reason"),
                "vllm_max_concurrency": nested_get(item, "metadata.vllm_max_concurrency"),
                "vllm_max_model_len": nested_get(item, "metadata.vllm_max_model_len"),
                "prompt_tok_s_agg": item.get("prompt_token_throughput_tok_s"),
                "prefill_tok_s_req": nested_get(item, "prefill_prompt_tok_s_approx.mean"),
                "decode_tok_s_agg": item.get("output_token_throughput_tok_s"),
                "total_tok_s": item.get("total_token_throughput_tok_s"),
                "per_req_e2e_tok_s": nested_get(item, "per_request_end_to_end_output_tok_s.mean"),
                "decode_tok_s_req": nested_get(item, "per_request_decode_output_tok_s.mean"),
                "benchmark_time_s": item.get("benchmark_time_s"),
                "lat_mean_s": nested_get(item, "latency_s.mean"),
                "lat_p90_s": nested_get(item, "latency_s.p90"),
                "ttft_mean_s": nested_get(item, "ttft_s.mean"),
                "ttft_p90_s": nested_get(item, "ttft_s.p90"),
                "sys_cpu_mean_pct": nested_get(resources, "system_cpu_pct.mean"),
                "sys_cpu_max_pct": nested_get(resources, "system_cpu_pct.max"),
                "sys_ram_mean_pct": nested_get(resources, "system_mem_used_pct.mean"),
                "sys_ram_max_pct": nested_get(resources, "system_mem_used_pct.max"),
                "swap_max_gib": bytes_to_gib(nested_get(resources, "system_swap_used_bytes.max")),
                "swap_max_pct": nested_get(resources, "system_swap_used_pct.max"),
                "proc_cpu_mean_pct": nested_get(resources, "process_cpu_pct_of_host.mean"),
                "proc_cpu_max_pct": nested_get(resources, "process_cpu_pct_of_host.max"),
                "proc_rss_max_gib": bytes_to_gib(nested_get(resources, "process_rss_bytes.max")),
                "gpu_util_mean_pct": nested_get(resources, "gpu_utilization_pct.mean_across_gpus_mean"),
                "gpu_util_max_pct": nested_get(resources, "gpu_utilization_pct.max_gpu_max"),
                "gpu_mem_max_gib": bytes_to_gib(nested_get(resources, "gpu_memory_used_bytes.max_total")),
                "gpu_mem_max_pct": nested_get(resources, "gpu_memory_used_pct.max_gpu_max"),
                "proc_gpu_mem_max_gib": bytes_to_gib(nested_get(resources, "process_gpu_memory_bytes.max")),
            }
        )
    return rows


def markdown_table(summaries: list[dict[str, Any]]) -> str:
    rows = table_rows(summaries)
    headers = [
        ("model", "Model"),
        ("mode", "Mode"),
        ("conc", "Conc"),
        ("req", "Req"),
        ("fail", "Fail"),
        ("aborted", "Abort"),
        ("vllm_max_concurrency", "VLLM max conc"),
        ("prompt_tok_s_agg", "Prompt tok/s agg"),
        ("prefill_tok_s_req", "Prefill tok/s req"),
        ("decode_tok_s_agg", "Decode tok/s agg"),
        ("decode_tok_s_req", "Decode tok/s req"),
        ("benchmark_time_s", "Benchmark time s"),
        ("total_tok_s", "Total tok/s"),
        ("lat_mean_s", "Lat mean s"),
        ("ttft_mean_s", "TTFT mean s"),
        ("gpu_util_mean_pct", "GPU util mean %"),
        ("gpu_util_max_pct", "GPU util max %"),
        ("gpu_mem_max_gib", "GPU mem GiB"),
        ("gpu_mem_max_pct", "GPU mem %"),
        ("proc_gpu_mem_max_gib", "Proc GPU GiB"),
        ("sys_cpu_mean_pct", "Host CPU mean %"),
        ("sys_cpu_max_pct", "Host CPU max %"),
        ("sys_ram_mean_pct", "Host RAM %"),
        ("swap_max_gib", "Swap GiB"),
        ("proc_cpu_mean_pct", "Proc CPU mean %"),
        ("proc_cpu_max_pct", "Proc CPU max %"),
        ("proc_rss_max_gib", "Proc RSS GiB"),
    ]
    lines = ["| " + " | ".join(label for _, label in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[key]) for key, _ in headers) + " |")
    return "\n".join(lines)


def write_outputs(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8"
    )
    table = markdown_table(summaries)
    (output_dir / "summary.md").write_text(table + "\n", encoding="utf-8")

    rows = table_rows(summaries)
    if rows:
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def parse_concurrencies(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise SystemExit("at least one concurrency is required")
    return values


def parse_suite_cases(raw: str | None) -> list[tuple[int, int]]:
    if not raw:
        return []
    cases = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise SystemExit(f"invalid suite case {item!r}; expected CONCURRENCY:REQUESTS")
        concurrency_raw, requests_raw = item.split(":", 1)
        concurrency = int(concurrency_raw)
        requests_count = int(requests_raw)
        if concurrency <= 0 or requests_count <= 0:
            raise SystemExit("suite case concurrency and requests must be positive")
        cases.append((concurrency, requests_count))
    return cases


def default_requests_for_concurrency(concurrency: int) -> int:
    return max(4, concurrency * 2)


def suite_cases(args: argparse.Namespace) -> list[tuple[int, int]]:
    explicit_cases = parse_suite_cases(args.suite_cases)
    if explicit_cases:
        return explicit_cases
    cases = [(1, 1), (1, 2)]
    for concurrency in parse_concurrencies(args.concurrencies):
        requests_count = args.suite_requests or default_requests_for_concurrency(concurrency)
        case = (concurrency, requests_count)
        if case not in cases:
            cases.append(case)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--concurrencies", default="1,2,4,8,16,32")
    parser.add_argument("--suite-cases", default=None, help="Explicit CONCURRENCY:REQUESTS pairs, e.g. 1:1,1:2,4:8,32:64")
    parser.add_argument("--suite-requests", type=int, default=None)
    parser.add_argument("--prompt-words", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--prompt-mode", choices=["unique", "repeated"], default="unique")
    parser.add_argument("--prompt-count", type=int, default=None)
    parser.add_argument("--no-per-request-details", action="store_true")
    parser.add_argument("--resource-interval", type=float, default=1.0)
    parser.add_argument("--process-match", default="vllm")
    parser.add_argument("--pid", type=int, action="append", default=[])
    parser.add_argument("--max-host-ram-pct", type=float, default=0.0)
    parser.add_argument("--max-swap-used-gib", type=float, default=4.0)
    parser.add_argument("--max-swap-growth-gib", type=float, default=1.0)
    parser.add_argument("--guard-grace-samples", type=int, default=2)
    parser.add_argument("--vllm-max-concurrency", type=float, default=None)
    parser.add_argument("--vllm-max-model-len", type=int, default=None)
    parser.add_argument("--concurrency-safety-fraction", type=float, default=0.85)
    parser.add_argument("--abort-suite-on-guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-config-path", default=None)
    args = parser.parse_args()

    run_config = None
    if args.run_config_path:
        run_config = json.loads(Path(args.run_config_path).read_text(encoding="utf-8"))

    if args.suite:
        summaries = []
        for concurrency, requests_count in suite_cases(args):
            print(
                f"running model={args.model} concurrency={concurrency} "
                f"requests={requests_count} prompt_mode={args.prompt_mode}",
                flush=True,
            )
            summary = run_benchmark(args, concurrency, requests_count)
            if run_config is not None:
                summary["run_config"] = run_config
            summaries.append(summary)
            if args.abort_suite_on_guard and summary.get("aborted"):
                print(f"aborting remaining suite cases: {summary.get('abort_reason')}", flush=True)
                break

        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path(args.output_dir or f"vllm-benchmark-{timestamp}")
        write_outputs(output_dir, summaries)
        print(markdown_table(summaries))
        print(f"\nwrote {output_dir / 'results.json'}")
        print(f"wrote {output_dir / 'summary.md'}")
        print(f"wrote {output_dir / 'summary.csv'}")
    else:
        summary = run_benchmark(args, args.concurrency, args.requests)
        if run_config is not None:
            summary["run_config"] = run_config
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
