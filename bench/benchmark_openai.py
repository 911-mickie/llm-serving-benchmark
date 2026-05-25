import argparse
import asyncio
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import httpx
import numpy as np
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values, p):
    if not values:
        return None
    return float(np.percentile(values, p))


def safe_mean(values):
    if not values:
        return None
    return float(statistics.mean(values))


def safe_stddev(values):
    if not values or len(values) < 2:
        return None
    return float(statistics.stdev(values))


def count_tokens(tokenizer, text):
    if tokenizer is None:
        return None
    return len(tokenizer.encode(text, add_special_tokens=False))


def count_input_tokens(tokenizer, messages):
    """Count tokens across all message contents."""
    if tokenizer is None:
        return None
    total = 0
    for msg in messages:
        total += len(tokenizer.encode(msg.get("content", ""), add_special_tokens=False))
    return total


def capture_gpu_snapshot():
    """Capture GPU memory state via nvidia-smi. Returns dict or None."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        gpus = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append({
                    "name": parts[0],
                    "driver_version": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_used_mb": int(parts[3]),
                    "memory_free_mb": int(parts[4]),
                    "utilization_pct": int(parts[5]),
                })
        return gpus if gpus else None
    except Exception:
        return None


async def run_one_request(client, base_url, model, row, tokenizer, temperature):
    url = base_url.rstrip("/") + "/v1/chat/completions"

    payload = {
        "model": model,
        "messages": row["messages"],
        "temperature": temperature,
        "max_tokens": row.get("max_tokens", 128),
        "stream": True,
    }

    started = time.perf_counter()
    first_token_at = None
    output_text = ""
    error = None

    try:
        async with client.stream("POST", url, json=payload, timeout=None) as response:
            if response.status_code != 200:
                body = await response.aread()
                return {
                    "id": row.get("id"),
                    "ok": False,
                    "status_code": response.status_code,
                    "error": body.decode("utf-8", errors="replace"),
                    "ttft_s": None,
                    "e2e_s": time.perf_counter() - started,
                    "input_tokens": count_input_tokens(tokenizer, row["messages"]),
                    "output_tokens": None,
                    "output_chars": 0,
                }

            async for line in response.aiter_lines():
                if not line:
                    continue
                data = line[len("data: "):].strip() if line.startswith("data: ") else line.strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    piece = delta.get("content", "")
                    if piece:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        output_text += piece
                except Exception:
                    continue

    except Exception as e:
        error = repr(e)

    finished = time.perf_counter()

    return {
        "id": row.get("id"),
        "ok": error is None,
        "status_code": 200 if error is None else None,
        "error": error,
        "ttft_s": None if first_token_at is None else first_token_at - started,
        "e2e_s": finished - started,
        "input_tokens": count_input_tokens(tokenizer, row["messages"]),
        "output_tokens": count_tokens(tokenizer, output_text),
        "output_chars": len(output_text),
        "text_preview": output_text[:200],
    }


async def run_single_pass(args, workload, tokenizer):
    """Run one complete benchmark pass. Returns list of result dicts."""
    semaphore = asyncio.Semaphore(args.concurrency)

    if args.warmup_requests > 0:
        print(f"  Warming up ({args.warmup_requests} requests)...")
        async with httpx.AsyncClient() as client:
            for i in range(args.warmup_requests):
                row = workload[i % len(workload)]
                await run_one_request(client, args.base_url, args.model, row, tokenizer, args.temperature)

    async with httpx.AsyncClient() as client:
        pbar = tqdm(total=args.num_requests, desc=f"  c={args.concurrency}")

        async def task(i):
            async with semaphore:
                row = workload[i % len(workload)]
                result = await run_one_request(client, args.base_url, args.model, row, tokenizer, args.temperature)
                pbar.update(1)
                return result

        bench_started = time.perf_counter()
        results = await asyncio.gather(*[task(i) for i in range(args.num_requests)])
        bench_finished = time.perf_counter()
        pbar.close()

    return results, bench_finished - bench_started


def build_summary(args, all_run_results, all_wall_times, gpu_before, gpu_after):
    """Aggregate across multiple runs into a single summary with variance."""
    all_ttfts = []
    all_e2es = []
    all_output_tokens = []
    all_input_tokens = []
    total_ok = 0
    total_failed = 0

    per_run_tok_s = []
    per_run_rps = []

    for results, wall_time in zip(all_run_results, all_wall_times):
        ok = [r for r in results if r["ok"]]
        failed = [r for r in results if not r["ok"]]
        total_ok += len(ok)
        total_failed += len(failed)

        ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
        e2es = [r["e2e_s"] for r in ok if r["e2e_s"] is not None]
        out_toks = [r["output_tokens"] for r in ok if r["output_tokens"] is not None]
        in_toks = [r["input_tokens"] for r in ok if r["input_tokens"] is not None]

        all_ttfts.extend(ttfts)
        all_e2es.extend(e2es)
        all_output_tokens.extend(out_toks)
        all_input_tokens.extend(in_toks)

        total_out_toks = sum(out_toks) if out_toks else None
        if total_out_toks and wall_time > 0:
            per_run_tok_s.append(total_out_toks / wall_time)
        if wall_time > 0:
            per_run_rps.append(len(ok) / wall_time)

    token_counting = "transformers tokenizer" if (args.tokenizer and AutoTokenizer is not None) else "disabled"

    return {
        "engine": args.engine,
        "base_url": args.base_url,
        "model": args.model,
        "dataset": args.dataset,
        "concurrency": args.concurrency,
        "num_requests_per_run": args.num_requests,
        "num_runs": args.runs,
        "warmup_requests": args.warmup_requests,
        "successful_requests": total_ok,
        "failed_requests": total_failed,
        # Throughput across runs
        "request_throughput_rps": safe_mean(per_run_rps),
        "request_throughput_rps_stddev": safe_stddev(per_run_rps),
        "output_tokens_per_second": safe_mean(per_run_tok_s),
        "output_tokens_per_second_stddev": safe_stddev(per_run_tok_s),
        # Token counts
        "output_tokens_total": sum(all_output_tokens) if all_output_tokens else None,
        "input_tokens_mean": safe_mean(all_input_tokens),
        # TTFT distribution (pooled across all runs)
        "ttft_s": {
            "mean": safe_mean(all_ttfts),
            "stddev": safe_stddev(all_ttfts),
            "p50": percentile(all_ttfts, 50),
            "p95": percentile(all_ttfts, 95),
            "p99": percentile(all_ttfts, 99),
        },
        # E2E distribution (pooled across all runs)
        "e2e_s": {
            "mean": safe_mean(all_e2es),
            "stddev": safe_stddev(all_e2es),
            "p50": percentile(all_e2es, 50),
            "p95": percentile(all_e2es, 95),
            "p99": percentile(all_e2es, 99),
        },
        # GPU memory snapshots
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        # Metadata
        "notes": {
            "streaming": True,
            "token_counting": token_counting,
            "temperature": args.temperature,
        },
    }


async def run_benchmark(args):
    workload = load_jsonl(args.dataset)

    tokenizer = None
    if args.tokenizer:
        if AutoTokenizer is None:
            print("[WARN] transformers not available. Token counting disabled.")
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    raw_path = Path(args.output_dir) / "raw_results.jsonl"
    summary_path = Path(args.output_dir) / "summary.json"

    print(f"\n[{args.engine}] dataset={args.dataset} concurrency={args.concurrency} runs={args.runs}")

    gpu_before = capture_gpu_snapshot()
    if gpu_before:
        print(f"  GPU before: {gpu_before[0]['name']} | used={gpu_before[0]['memory_used_mb']}MB / {gpu_before[0]['memory_total_mb']}MB")

    all_run_results = []
    all_wall_times = []

    for run_idx in range(args.runs):
        if args.runs > 1:
            print(f"Run {run_idx + 1}/{args.runs}")
        results, wall_time = await run_single_pass(args, workload, tokenizer)
        all_run_results.append(results)
        all_wall_times.append(wall_time)

    gpu_after = capture_gpu_snapshot()
    if gpu_after:
        print(f"  GPU after:  {gpu_after[0]['name']} | used={gpu_after[0]['memory_used_mb']}MB / {gpu_after[0]['memory_total_mb']}MB")

    # Write raw results (all runs flattened)
    with open(raw_path, "w", encoding="utf-8") as f:
        for run_idx, results in enumerate(all_run_results):
            for r in results:
                r["run"] = run_idx
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = build_summary(args, all_run_results, all_wall_times, gpu_before, gpu_after)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary:")
    print(f"  Throughput: {summary['output_tokens_per_second']:.1f} tok/s")
    print(f"  TTFT p95:   {summary['ttft_s']['p95']*1000:.1f} ms")
    print(f"  E2E  p95:   {summary['e2e_s']['p95']:.3f} s")
    print(f"  Failed:     {summary['failed_requests']}")
    print(f"  Saved to:   {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--num-requests", type=int, default=50)
    parser.add_argument("--runs", type=int, default=3, help="Number of independent runs (default: 3)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup-requests", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_benchmark(parse_args()))