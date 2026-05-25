# llm-serving-benchmark

Benchmarking vLLM vs SGLang on Qwen2.5-7B-Instruct (FP16 and AWQ) on a single RTX 3090.

---

## Motivation

Before committing to an inference engine for a self-hosted setup, I wanted real numbers on consumer GPU hardware — not cloud benchmarks, not synthetic toy servers. This lab runs a controlled comparison between the two dominant open-source serving engines across both a normal-precision model and its AWQ-quantized counterpart, using the same async benchmark harness and the same workloads for each run.

---

## Hardware & Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 3090 (24 GB VRAM) |
| Driver | 570.86.15 |
| CUDA | 12.8 |
| vLLM image | `vllm/vllm-openai:v0.8.5` |
| SGLang image | `lmsysorg/sglang:v0.4.9.post6-cu126` |
| SGLang AWQ image | `sglang-awq-vllmops:local` (built from `docker/Dockerfile.sglang-awq`) |

`vllm:latest` was intentionally avoided — it required CUDA 13.x which the driver stack did not satisfy. The SGLang AWQ path required a custom image to resolve dependency conflicts between SGLang, vLLM ops, and transformers versions.

---

## Experiment Matrix

| Model | Engine | Workloads | Concurrency levels |
|---|---|---|---|
| Qwen2.5-7B-Instruct (FP16) | vLLM | chat_trace, prefix_heavy | 1, 4 |
| Qwen2.5-7B-Instruct (FP16) | SGLang | chat_trace, prefix_heavy | 1, 4 |
| Qwen2.5-7B-Instruct-AWQ | vLLM | chat_trace, prefix_heavy | 1, 2, 4 |
| Qwen2.5-7B-Instruct-AWQ | SGLang | chat_trace, prefix_heavy | 1, 2, 4 |

**Workloads:**
- `chat_trace.jsonl` — 20 short-prefix conversational requests with varied topics
- `prefix_heavy.jsonl` — ~20 requests sharing a long common prefix, designed to stress prefix caching

One engine ran at a time. Each run included 4 warmup requests before measurements were recorded.

---

## Results

### FP16 — Qwen2.5-7B-Instruct

| Engine | Dataset | Concurrency | Req/s | Tok/s | TTFT p95 (s) | E2E p95 (s) | Failed |
|---|---|---:|---:|---:|---:|---:|---:|
| vllm | chat_trace | 1 | 0.306 | 49.22 | 0.027 | 4.067 | 0 |
| sglang | chat_trace | 1 | 0.306 | 49.38 | 0.043 | 4.052 | 0 |
| vllm | chat_trace | 4 | 1.218 | 186.16 | 0.042 | 4.185 | 0 |
| sglang | chat_trace | 4 | 1.203 | 183.72 | 0.062 | 4.229 | 0 |
| vllm | prefix_heavy | 1 | 0.316 | 48.37 | 0.028 | 3.700 | 0 |
| sglang | prefix_heavy | 1 | 0.319 | 48.94 | 0.043 | 3.654 | 0 |
| vllm | prefix_heavy | 4 | 1.188 | 179.27 | 0.043 | 3.826 | 0 |
| sglang | prefix_heavy | 4 | 1.198 | 181.90 | 0.062 | 3.786 | 0 |

### AWQ — Qwen2.5-7B-Instruct-AWQ

| Engine | Dataset | Concurrency | Req/s | Tok/s | TTFT p95 (s) | E2E p95 (s) | Failed |
|---|---|---:|---:|---:|---:|---:|---:|
| vllm | chat_trace | 1 | 0.795 | 128.18 | 0.020 | 1.561 | 0 |
| sglang | chat_trace | 1 | 0.927 | 148.87 | 0.022 | 1.341 | 0 |
| vllm | chat_trace | 2 | 1.571 | 242.44 | 0.019 | 1.622 | 0 |
| sglang | chat_trace | 2 | 1.828 | 281.58 | 0.028 | 1.399 | 0 |
| vllm | chat_trace | 4 | 3.047 | 467.56 | 0.028 | 1.668 | 0 |
| sglang | chat_trace | 4 | 3.513 | 537.30 | 0.034 | 1.449 | 0 |
| vllm | prefix_heavy | 1 | 0.864 | 122.40 | 0.021 | 1.454 | 0 |
| sglang | prefix_heavy | 1 | 0.994 | 145.69 | 0.022 | 1.226 | 0 |
| vllm | prefix_heavy | 2 | 1.694 | 235.98 | 0.019 | 1.503 | 0 |
| sglang | prefix_heavy | 2 | 1.960 | 277.31 | 0.028 | 1.282 | 0 |
| vllm | prefix_heavy | 4 | 3.282 | 450.42 | 0.029 | 1.533 | 0 |
| sglang | prefix_heavy | 4 | 3.696 | 520.01 | 0.036 | 1.309 | 0 |

---

## Key Findings

- **FP16: both engines are effectively equivalent.** Throughput within 2% of each other at every concurrency level. vLLM has meaningfully lower TTFT p95 (0.027s vs 0.043s at c1), but E2E latency is nearly identical. If you are running FP16, engine choice barely matters on this hardware.

- **AWQ: SGLang consistently outperforms vLLM by ~15%.** At concurrency 4, SGLang delivers 3.51 req/s vs vLLM's 3.05 req/s on chat_trace, and 537 tok/s vs 468 tok/s. E2E p95 latency is also lower (1.449s vs 1.668s). vLLM retains a slight TTFT advantage even in AWQ mode, but SGLang wins on every throughput metric.

- **AWQ delivers roughly 3× the throughput of FP16.** On the same hardware, chat_trace at concurrency 4 goes from 1.2 req/s (FP16) to 3.5 req/s (SGLang AWQ). This is not a small tuning improvement — it is a different operating point entirely.

- **AWQ cuts E2E latency by ~65%.** FP16 E2E p95 was ~4s. AWQ brought it down to 1.3–1.7s. For latency-sensitive use cases on consumer GPUs, quantization matters far more than engine choice.

- **Zero failures across all 20 benchmark cells.** Both engines handled all workloads without a single failed request.

---

## Repo Structure

```
llm-serving-lab-final/
│
├── docker-compose.yml              # vLLM server (parameterised via env file)
├── docker-compose.sglang.yml       # SGLang server (parameterised via env file)
│
├── docker/
│   └── Dockerfile.sglang-awq       # Custom SGLang image that resolves AWQ dependency conflicts
│
├── bench/
│   ├── benchmark_openai.py         # Async benchmark client (OpenAI /v1/completions)
│   ├── aggregate_results.py        # Post-run CSV + Markdown table generator
│   ├── Dockerfile.bench            # Benchmark runner image (python:3.11-slim)
│   ├── run_baseline.sh             # OLD: early vLLM built-in bench tool (not the final harness)
│   ├── run_concurrency_sweep.sh    # OLD: early concurrency sweep script
│   └── run_sglang_smoke.sh         # OLD: early SGLang smoke test
│
├── datasets/
│   ├── chat_trace.jsonl            # Short-prefix conversational requests
│   └── prefix_heavy.jsonl          # Long-shared-prefix requests (prefix cache stress)
│
├── scripts/
│   ├── healthcheck.sh              # curl /v1/models and print response
│   ├── healthcheck_sglang.sh       # Check /health, /v1/models, run a smoke completion
│   ├── run_qwen_baseline_matrix.sh         # Full FP16 matrix: vLLM + SGLang, c1/c4
│   ├── run_qwen_awq_concurrency_matrix.sh  # Full AWQ matrix: vLLM + SGLang, c1/c2/c4
│   └── run_qwen_awq_sglang_only_matrix.sh  # SGLang-only partial run (for resuming)
│
├── results/
│   ├── qwen-baseline-engine-comparison/    # FP16 results (summary.csv, summary.md, per-run JSONs)
│   ├── qwen-awq-engine-comparison/         # AWQ results (same structure)
│   └── old_results/                        # Earlier pilot runs (Cohere model, not the final study)
│
├── docs/
│   └── checkpoint_04_qwen_base_engine_comparison.md  # Narrative writeup for the FP16 run
│
├── .env.example                    # Template showing all required env variables
└── requirements-bench.txt          # Benchmark client dependencies (pinned)
```

---

## How to Reproduce

### Prerequisites

- Docker with NVIDIA Container Toolkit (`nvidia-container-toolkit`)
- `docker compose` v2
- A HuggingFace account with a read token (for downloading Qwen2.5 weights)
- An NVIDIA GPU with at least 16 GB VRAM (24 GB used here)

### 1. Configure environment files

```bash
cp .env.example .env
# Edit .env and set HF_TOKEN and HF_CACHE to your values
```

Create variants for each engine/model combination:

```bash
cp .env.example .env.qwen.vllm       # for FP16 vLLM
cp .env.example .env.qwen.sglang     # for FP16 SGLang
cp .env.example .env.qwen-awq.vllm   # for AWQ vLLM
cp .env.example .env.qwen-awq.sglang # for AWQ SGLang
```

Set `MODEL_NAME` and `SGLANG_IMAGE`/`VLLM_IMAGE` appropriately in each file. See `.env.example` for all available variables and the values used in the final experiment.

### 2. Pull serving images

```bash
docker pull vllm/vllm-openai:v0.8.5
docker pull lmsysorg/sglang:v0.4.9.post6-cu126
```

### 3. Build the custom SGLang AWQ image

Required for the AWQ path only:

```bash
docker build --no-cache \
  -f docker/Dockerfile.sglang-awq \
  -t sglang-awq-vllmops:local \
  .
```

### 4. Build the benchmark runner image

```bash
docker build -f bench/Dockerfile.bench -t llm-serving-bench:local .
```

### 5. Run the FP16 matrix

```bash
bash scripts/run_qwen_baseline_matrix.sh
```

Starts vLLM, benchmarks it, stops it, starts SGLang, benchmarks it, stops it, then aggregates results to `results/qwen-baseline-engine-comparison/summary.csv`.

### 6. Run the AWQ matrix

```bash
bash scripts/run_qwen_awq_concurrency_matrix.sh
```

Same structure, using the AWQ model and custom SGLang image. Results go to `results/qwen-awq-engine-comparison/summary.csv`.

### 7. Health-check a running server manually

```bash
bash scripts/healthcheck.sh                    # vLLM (port 8001)
bash scripts/healthcheck_sglang.sh             # SGLang (port 30000)
```

---

## Limitations

- **Synthetic datasets.** Both workloads are hand-crafted JSONL files (20–39 requests each), not real traffic traces. Results reflect the specific input/output length distributions in those files.
- **Max concurrency 4.** The server was configured with `MAX_NUM_SEQS=4` / `SGLANG_MAX_RUNNING_REQUESTS=4` to fit the 24 GB GPU. Engines with higher concurrency may behave differently relative to each other.
- **No output quality evaluation.** Only throughput and latency were measured. AWQ quantization accuracy loss is not assessed here.
- **Single GPU, single node.** Multi-GPU or tensor-parallel results would likely differ, especially for SGLang which has deeper TP support.
- **FP16 and AWQ runs were conducted on different days.** Background GPU state (e.g., cached model weights) may introduce minor variance between the two experiment sets.

---

## What's Next

If continuing this work: higher concurrency sweeps (c8, c16) to find the saturation point, real traffic traces from public datasets, speculative decoding comparison, and a multi-GPU run with tensor parallelism across both engines.

---

## License

MIT
