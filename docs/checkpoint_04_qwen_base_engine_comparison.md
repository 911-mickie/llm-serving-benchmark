# Checkpoint 04 — Qwen2.5 Base Engine Comparison

## Setup

- GPU: RTX 3090
- Model: Qwen/Qwen2.5-7B-Instruct
- Engines: vLLM and SGLang
- Workloads:
  - chat_trace
  - prefix_heavy
- Concurrency levels:
  - 1
  - 4
- Token counting: transformers tokenizer
- Serving constraint: one engine active at a time

## Result Summary

| Engine | Dataset | Concurrency | Req/s | Tok/s | TTFT p95 (s) | E2E p95 (s) | Failed |
|---|---|---:|---:|---:|---:|---:|---:|
| sglang | chat_trace | 1 | 0.306 | 49.38 | 0.043 | 4.052 | 0 |
| vllm | chat_trace | 1 | 0.306 | 49.22 | 0.027 | 4.067 | 0 |
| sglang | chat_trace | 4 | 1.203 | 183.72 | 0.062 | 4.229 | 0 |
| vllm | chat_trace | 4 | 1.218 | 186.16 | 0.042 | 4.185 | 0 |
| sglang | prefix_heavy | 1 | 0.319 | 48.94 | 0.043 | 3.654 | 0 |
| vllm | prefix_heavy | 1 | 0.316 | 48.37 | 0.028 | 3.700 | 0 |
| sglang | prefix_heavy | 4 | 1.198 | 181.90 | 0.062 | 3.786 | 0 |
| vllm | prefix_heavy | 4 | 1.188 | 179.27 | 0.043 | 3.826 | 0 |

## Interpretation

vLLM consistently achieved lower p95 TTFT across both workloads and both concurrency levels.

Throughput was very close between engines. vLLM was slightly ahead on chat_trace at concurrency 4, while SGLang was slightly ahead on prefix_heavy at concurrency 4. The differences were small enough that this result should not be framed as a decisive throughput win for either engine.

E2E p95 latency was also close. The clearest base-model finding is that vLLM had better first-token responsiveness, while total request completion latency and token throughput were broadly comparable.

## Current Finding

For Qwen2.5-7B-Instruct base/FP16 on RTX 3090, vLLM provides lower first-token latency, while vLLM and SGLang are nearly tied on throughput and end-to-end latency for this small workload/concurrency matrix.

## Limitations

- Only base/FP16 model tested.
- No AWQ comparison yet.
- Only concurrency 1 and 4 tested.
- No quality evaluation yet.
- No profiler evidence yet.
- Prefix-heavy workload did not show a large engine separation in this run.
