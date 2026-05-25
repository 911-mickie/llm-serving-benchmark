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
