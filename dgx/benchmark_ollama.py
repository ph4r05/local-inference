#!/usr/bin/env python3
import argparse
import concurrent.futures
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_vllm_openai as common  # noqa: E402


def parse_ollama_stream_line(line: bytes) -> dict[str, Any] | None:
    raw = line.strip()
    if not raw:
        return None
    return json.loads(raw)


def run_one(
    request_id: int,
    prompt_id: int,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    num_ctx: int,
) -> common.Result:
    url = f"{base_url.rstrip('/')}/api/generate"
    body = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": num_ctx,
        },
    }

    start = time.perf_counter()
    first_token_at = None
    prompt_tokens = None
    total_tokens = None
    output_tokens = 0

    try:
        with requests.post(url, json=body, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                event = parse_ollama_stream_line(line)
                if not event:
                    continue

                chunk = event.get("response")
                if chunk:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()

                if event.get("done"):
                    prompt_tokens = event.get("prompt_eval_count")
                    output_tokens = event.get("eval_count", output_tokens) or output_tokens
                    total_tokens = (prompt_tokens or 0) + output_tokens if output_tokens is not None else None
                    break
    except Exception as exc:
        return common.Result(
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
    if prompt_tokens is None:
        prompt_tokens = 0
    if total_tokens is None:
        total_tokens = prompt_tokens + output_tokens

    return common.Result(
        request_id=request_id,
        prompt_id=prompt_id,
        ok=True,
        latency_s=end - start,
        ttft_s=None if first_token_at is None else first_token_at - start,
        output_tokens=output_tokens,
        prompt_tokens=prompt_tokens,
        total_tokens=total_tokens,
    )


def summarize(
    results: list[common.Result],
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
    return common.summarize(
        results,
        wall_s,
        model,
        base_url,
        concurrency,
        prompt_words,
        max_tokens,
        prompt_mode,
        prompt_count,
        include_per_request,
        resources,
        metadata,
    )


def run_benchmark(args: argparse.Namespace, concurrency: int, requests_count: int) -> dict[str, Any]:
    prompt_count = args.prompt_count or requests_count
    prompts = common.make_prompts(args.prompt_words, prompt_count, args.prompt_mode)

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
            args.context_length,
        )
        if not warmup.ok:
            raise SystemExit(f"warmup failed: {warmup.error}")

    monitor = common.ResourceMonitor(
        args.resource_interval,
        args.process_match,
        args.pid,
        max_host_ram_pct=args.max_host_ram_pct if args.max_host_ram_pct and args.max_host_ram_pct > 0 else None,
        max_swap_used_gib=args.max_swap_used_gib,
        max_swap_growth_gib=args.max_swap_growth_gib,
        guard_grace_samples=args.guard_grace_samples,
    )
    monitor.start()
    started = time.perf_counter()
    results: list[common.Result] = []
    submitted: set[int] = set()
    completed_ids: set[int] = set()
    next_request_id = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures: dict[concurrent.futures.Future[common.Result], int] = {}

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
                        args.context_length,
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
                            common.Result(
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

            for future in done:
                request_id = futures.pop(future)
                completed_ids.add(request_id)
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(
                        common.Result(
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
                    common.Result(
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
        "vllm_max_concurrency": args.ollama_num_parallel,
        "vllm_max_model_len": args.context_length,
        "concurrency_safety_fraction": args.concurrency_safety_fraction,
        "host_ram_guard_pct": args.max_host_ram_pct,
        "swap_used_guard_gib": args.max_swap_used_gib,
        "swap_growth_guard_gib": args.max_swap_growth_gib,
        "backend": "ollama",
        "ollama_num_parallel": args.ollama_num_parallel,
        "ollama_context_length": args.context_length,
        "ollama_max_queue": args.ollama_max_queue,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--concurrencies", default="1,2,4,8,16,32")
    parser.add_argument("--suite-cases", default=None)
    parser.add_argument("--suite-requests", type=int, default=None)
    parser.add_argument("--prompt-words", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--prompt-mode", choices=["unique", "repeated"], default="unique")
    parser.add_argument("--prompt-count", type=int, default=None)
    parser.add_argument("--no-per-request-details", action="store_true")
    parser.add_argument("--resource-interval", type=float, default=1.0)
    parser.add_argument("--process-match", default="ollama serve")
    parser.add_argument("--pid", type=int, action="append", default=[])
    parser.add_argument("--max-host-ram-pct", type=float, default=0.0)
    parser.add_argument("--max-swap-used-gib", type=float, default=4.0)
    parser.add_argument("--max-swap-growth-gib", type=float, default=1.0)
    parser.add_argument("--guard-grace-samples", type=int, default=2)
    parser.add_argument("--concurrency-safety-fraction", type=float, default=1.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--ollama-num-parallel", type=int, default=1)
    parser.add_argument("--ollama-max-queue", type=int, default=512)
    args = parser.parse_args()

    if args.suite:
        summaries = []
        for concurrency, requests_count in common.suite_cases(args):
            print(
                f"running model={args.model} concurrency={concurrency} requests={requests_count} prompt_mode={args.prompt_mode}",
                flush=True,
            )
            summary = run_benchmark(args, concurrency, requests_count)
            summaries.append(summary)
            if summary.get("aborted"):
                print(f"aborting remaining suite cases: {summary.get('abort_reason')}", flush=True)
                break

        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path(args.output_dir or f"ollama-benchmark-{timestamp}")
        common.write_outputs(output_dir, summaries)
        print(common.markdown_table(summaries))
        print(f"\nwrote {output_dir / 'results.json'}")
        print(f"wrote {output_dir / 'summary.md'}")
        print(f"wrote {output_dir / 'summary.csv'}")
    else:
        summary = run_benchmark(args, args.concurrency, args.requests)
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
