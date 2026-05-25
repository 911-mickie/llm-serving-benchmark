import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    rows = []

    for path in sorted(Path(args.results_dir).rglob("summary.json")):
        with open(path, "r", encoding="utf-8") as f:
            s = json.load(f)

        # Skip invalid/non-tokenized runs
        if s.get("notes", {}).get("token_counting") != "transformers tokenizer":
            continue

        rows.append({
            "engine": s.get("engine"),
            "dataset": Path(s.get("dataset", "")).stem,
            "concurrency": s.get("concurrency"),
            "warmup_requests": s.get("warmup_requests"),
            "successful_requests": s.get("successful_requests"),
            "failed_requests": s.get("failed_requests"),
            "request_throughput_rps": s.get("request_throughput_rps"),
            "output_tokens_per_second": s.get("output_tokens_per_second"),
            "ttft_p50_s": s.get("ttft_s", {}).get("p50"),
            "ttft_p95_s": s.get("ttft_s", {}).get("p95"),
            "ttft_p99_s": s.get("ttft_s", {}).get("p99"),
            "e2e_p50_s": s.get("e2e_s", {}).get("p50"),
            "e2e_p95_s": s.get("e2e_s", {}).get("p95"),
            "e2e_p99_s": s.get("e2e_s", {}).get("p99"),
            "path": str(path),
        })

    rows.sort(key=lambda r: (r["dataset"], r["concurrency"], r["engine"]))

    fieldnames = [
        "engine",
        "dataset",
        "concurrency",
        "warmup_requests",
        "successful_requests",
        "failed_requests",
        "request_throughput_rps",
        "output_tokens_per_second",
        "ttft_p50_s",
        "ttft_p95_s",
        "ttft_p99_s",
        "e2e_p50_s",
        "e2e_p95_s",
        "e2e_p99_s",
        "path",
    ]

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("| Engine | Dataset | Concurrency | Req/s | Tok/s | TTFT p95 (s) | E2E p95 (s) | Failed |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['engine']} "
                f"| {r['dataset']} "
                f"| {r['concurrency']} "
                f"| {r['request_throughput_rps']:.3f} "
                f"| {r['output_tokens_per_second']:.2f} "
                f"| {r['ttft_p95_s']:.3f} "
                f"| {r['e2e_p95_s']:.3f} "
                f"| {r['failed_requests']} |\n"
            )

    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")
    print(f"Included {len(rows)} tokenized result files")


if __name__ == "__main__":
    main()
