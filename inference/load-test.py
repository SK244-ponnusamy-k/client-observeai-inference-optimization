"""
inference/load-test.py

Local load test for vLLM — measures TTFT, latency, throughput.
Runs from your laptop via port-forward.

Usage:
    pip install openai
    kubectl port-forward svc/vllm-inference 8080:8000 &
    python inference/load-test.py --model qwen-0.5b
    python inference/load-test.py --model gpt-oss-20b --requests 20 --port 8080
"""

import argparse
import time
import statistics
from openai import OpenAI

PROMPTS = [
    "Explain Kubernetes in two sentences.",
    "What is a container?",
    "What is machine learning?",
    "Explain REST API briefly.",
    "What is cloud computing?",
    "What is GPU inference?",
    "Explain transformer models briefly.",
    "What is vector embedding?",
]

def run(model: str, port: int, num_requests: int) -> None:
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")

    # Check health
    import urllib.request
    try:
        urllib.request.urlopen(f"http://localhost:{port}/health")
    except Exception:
        print(f"[ERROR] vLLM not reachable at localhost:{port}")
        print("Start port-forward: kubectl port-forward svc/vllm-inference 8080:8000 &")
        raise SystemExit(1)

    print(f"\n{'═'*56}")
    print(f"  LOAD TEST — {model}")
    print(f"  Endpoint : http://localhost:{port}")
    print(f"  Requests : {num_requests}")
    print(f"{'═'*56}\n")

    ttfts, totals, token_counts = [], [], []

    for i in range(num_requests):
        prompt = PROMPTS[i % len(PROMPTS)]
        start = time.time()
        first_token_time = None
        content = ""

        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                if first_token_time is None:
                    first_token_time = time.time()
                content += delta

        end = time.time()
        ttft  = (first_token_time - start) * 1000 if first_token_time else 0
        total = (end - start) * 1000
        words = len(content.split())

        ttfts.append(ttft)
        totals.append(total)
        token_counts.append(words)

        print(f"[{i+1:3d}/{num_requests}] TTFT: {ttft:6.1f}ms | Total: {total:6.1f}ms | ~{words} words")

    # Summary
    s_ttft  = sorted(ttfts)
    s_total = sorted(totals)
    p = lambda lst, pct: lst[max(0, int(len(lst) * pct / 100) - 1)]

    print(f"\n{'═'*56}")
    print(f"  RESULTS ({num_requests} requests, sequential)")
    print(f"{'═'*56}")
    print(f"  TTFT avg        : {statistics.mean(ttfts):.1f} ms")
    print(f"  TTFT p50        : {statistics.median(ttfts):.1f} ms")
    print(f"  TTFT p90        : {p(s_ttft, 90):.1f} ms")
    print(f"  TTFT p99        : {p(s_ttft, 99):.1f} ms")
    print(f"  TTFT min/max    : {min(ttfts):.1f} / {max(ttfts):.1f} ms")
    print(f"  Latency avg     : {statistics.mean(totals):.1f} ms")
    print(f"  Latency p50     : {statistics.median(totals):.1f} ms")
    print(f"  Latency p90     : {p(s_total, 90):.1f} ms")
    print(f"  Latency p99     : {p(s_total, 99):.1f} ms")
    print(f"  Avg words/resp  : {statistics.mean(token_counts):.1f}")
    print(f"  Total words     : {sum(token_counts)}")
    print(f"{'═'*56}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM load test")
    parser.add_argument("--model",    default="qwen-0.5b", help="Model name")
    parser.add_argument("--port",     type=int, default=8080, help="Port-forward port")
    parser.add_argument("--requests", type=int, default=10,   help="Number of requests")
    args = parser.parse_args()
    run(args.model, args.port, args.requests)
