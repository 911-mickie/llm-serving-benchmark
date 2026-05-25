| Engine | Dataset | Concurrency | Req/s | Tok/s | TTFT p95 (s) | E2E p95 (s) | Failed |
|---|---|---:|---:|---:|---:|---:|---:|
| sglang | chat_trace | 1 | 0.927 | 148.87 | 0.022 | 1.341 | 0 |
| vllm | chat_trace | 1 | 0.795 | 128.18 | 0.020 | 1.561 | 0 |
| sglang | chat_trace | 2 | 1.828 | 281.58 | 0.028 | 1.399 | 0 |
| vllm | chat_trace | 2 | 1.571 | 242.44 | 0.019 | 1.622 | 0 |
| sglang | chat_trace | 4 | 3.513 | 537.30 | 0.034 | 1.449 | 0 |
| vllm | chat_trace | 4 | 3.047 | 467.56 | 0.028 | 1.668 | 0 |
| sglang | prefix_heavy | 1 | 0.994 | 145.69 | 0.022 | 1.226 | 0 |
| vllm | prefix_heavy | 1 | 0.864 | 122.40 | 0.021 | 1.454 | 0 |
| sglang | prefix_heavy | 2 | 1.960 | 277.31 | 0.028 | 1.282 | 0 |
| vllm | prefix_heavy | 2 | 1.694 | 235.98 | 0.019 | 1.503 | 0 |
| sglang | prefix_heavy | 4 | 3.696 | 520.01 | 0.036 | 1.309 | 0 |
| vllm | prefix_heavy | 4 | 3.282 | 450.42 | 0.029 | 1.533 | 0 |
