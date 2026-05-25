# Engineering Challenges & Lessons Learned

This document covers the real infrastructure problems encountered while building this benchmark lab. The experiment did not start clean — it started with a working vLLM pilot and grew into a controlled multi-engine comparison by systematically resolving blockers. What follows is that journey.

---

## The Starting Point

The project began as a straightforward vLLM serving test using `CohereLabs/c4ai-command-r7b-12-2024`. The goal was simple: get a model running behind an OpenAI-compatible endpoint and benchmark it. That part worked. But the moment the scope expanded to a proper engine comparison with quantization, the real engineering work began.

---

## Challenge 1: CUDA/Driver Compatibility — Why `latest` Is a Lie

The first real blocker appeared immediately when trying to move to a more current vLLM image.

```text
nvidia-container-cli: requirement error: unsatisfied condition: cuda>=13.0
```

The container refused to start. Not because of a model problem, not because of a Python dependency — but because the Docker image silently required a newer CUDA stack than the host GPU driver could provide. The RTX 3090 was running driver `570.x` with CUDA 12.8. The `latest` tag had quietly moved to CUDA 13.x requirements.

**The fix was simple once the root cause was clear:** pin the image to a known-compatible version.

```bash
docker pull vllm/vllm-openai:v0.8.5
```

**The lesson was not simple:** in ML infrastructure, `latest` is not a convenience — it is a liability. Every serving image in this project was pinned from that point forward, and the exact tags are documented in the README. This is one of those errors that feels like black magic until you understand that Docker image tags are essentially dependency contracts with no enforcement mechanism.

---

## Challenge 2: The Cohere Model Was the Wrong Tool for the Job

After the first vLLM baseline was working, the project hit a more fundamental problem. The plan was to compare FP16 vs AWQ quantized serving. But `CohereLabs/c4ai-command-r7b-12-2024` had no clean official AWQ counterpart.

This was a methodology problem, not a technical one. Using a third-party quantized conversion from a different source would have invalidated the comparison — any performance difference could have been attributed to the quantization process itself rather than the serving engine.

**The fix was to change the model family entirely** to `Qwen/Qwen2.5-7B-Instruct`, which has a clean, officially maintained AWQ variant (`Qwen/Qwen2.5-7B-Instruct-AWQ`) from the same architecture. This meant the only variable between the FP16 and AWQ runs was the precision format — exactly what a controlled experiment requires.

This was the most consequential decision in the project. It meant throwing away the existing pilot results and restarting the baseline, but it made the final comparison scientifically defensible.

**Lesson:** model selection should be driven by the experiment design, not by familiarity or convenience. For a quantization benchmark, the FP16 and AWQ variants must come from the same family or the results are not comparable.

---

## Challenge 3: SGLang AWQ Required a Custom Docker Image

Once the Qwen2.5 baseline was working on both engines, the AWQ path introduced a new class of problem: dependency conflicts inside the SGLang container.

The standard `lmsysorg/sglang:v0.4.9.post6-cu126` image shipped with versions of `vllm` and `transformers` that were incompatible with the AWQ kernel path needed for `Qwen2.5-7B-Instruct-AWQ`. Simply swapping the model name and running the same container did not work.

**The solution was to build a custom image** (`docker/Dockerfile.sglang-awq`) that:
- starts from the base SGLang image
- uninstalls the conflicting `vllm` and `transformers` versions
- pins `vllm==0.9.0.1` and `transformers==4.53.3`
- runs a sanity import check at build time to catch silent failures early

```bash
docker build --no-cache \
  -f docker/Dockerfile.sglang-awq \
  -t sglang-awq-vllmops:local .
```

This is the kind of problem that does not appear in tutorials. It only appears when you try to serve a quantized model on hardware you actually own, with software that was not written to make this easy. The custom Dockerfile is in the repo because it is part of the reproducibility story — anyone trying to replicate the AWQ results needs it.

**Lesson:** quantized serving is a separate deployment path, not a model-name swap. Treat it as its own stack with its own image, its own validation, and its own documentation.

---

## Challenge 4: GPU Memory Contention on a Single-Machine Setup

Running two inference engines on one GPU introduced an operational risk that is easy to underestimate: a stopped container is not always a dead container. Orphaned containers from earlier runs could still hold VRAM or occupy ports, causing the next engine to fail during model loading — sometimes silently.

The symptom was not always a clean error. Sometimes the new serving container would start, pass the health check, and then either hang during model loading or crash with a CUDA out-of-memory error that looked unrelated to the real cause.

**The fix was process, not code.** Every engine switch required:

```bash
# verify what's actually running
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
nvidia-smi

# explicitly tear down before starting the next engine
docker compose -f docker-compose.yml down
```

This discipline was eventually encoded into the matrix runner scripts themselves — each script explicitly stops the previous engine before starting the next one, with a 5-second rest between runs.

**Lesson:** on a single GPU, `docker ps` and `nvidia-smi` are not optional steps. They are the pre-flight check. The matrix scripts enforce this automatically so that human memory does not have to.

---

## Challenge 5: Manual Workflow → Scripted Benchmark Matrix

The early experiment was run command by command: start server, health check, run one workload, run another, stop server, switch engine, repeat. After several iterations this became error-prone in ways that would quietly corrupt results — wrong model served, wrong dataset used, results saved to the wrong folder, no cooldown between runs.

The solution was to move the entire workflow into parameterised shell scripts (`scripts/run_qwen_baseline_matrix.sh`, `scripts/run_qwen_awq_concurrency_matrix.sh`) that encode the full sequence: server start → health check → warmup → benchmark across all concurrency levels → server stop → next engine. Results are written to a structured path:

```text
results/<experiment>/<engine>/<model>/<dataset>/c<concurrency>/summary.json
```

This structure was designed to make the aggregation script's job trivial and to make it impossible to accidentally mix results from different runs.

**Lesson:** once an experiment has more than three manual steps, it should be scripted. Manual commands are fine for discovery. They are not acceptable for final reproducible benchmarks.

---

## What These Challenges Add Up To

These were not random problems. They followed a pattern common to real ML infrastructure work:

| Category | Problem |
|---|---|
| Container/runtime compatibility | CUDA version mismatch, `latest` tag instability |
| Experiment design | Model selection invalidating quantization comparison |
| Dependency management | AWQ serving requiring custom image build |
| Resource contention | Single GPU, multiple engines, orphaned containers |
| Reproducibility | Ad-hoc commands → scripted parameterised matrix |

Solving each one made the final results more trustworthy, not just more convenient. The benchmark numbers in this repo mean something because the infrastructure underneath them was built carefully — and these were the problems that forced that care.