"""BatchServe Benchmarking Suite.

Simulates concurrent client requests against BatchServe to measure throughput
(tokens/sec), Time-To-First-Token (TTFT), Inter-Token Latency (ITL), and
Paged KV-cache memory utilization under continuous batch scheduling.
"""

import argparse
import asyncio
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List, Optional
import httpx

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROMPT_BANK = [
    "Explain the concept of continuous batching in high-throughput LLM inference systems.",
    "How does virtual memory paging in operating systems compare to PagedAttention in vLLM?",
    "Describe the difference between external fragmentation and internal memory fragmentation.",
    "Why does static padding in naive LLM batching lead to severe GPU memory waste?",
    "Explain how iteration-level scheduling prevents head-of-line blocking during generation.",
    "What is nucleus sampling (top-p) and how does it balance diversity with text coherence?",
    "Discuss the trade-offs between preemption by recomputation versus swapping to CPU memory.",
    "How does prefix caching reduce prompt processing latency for common system instructions?",
]


async def send_generate_request(
    client: httpx.AsyncClient,
    base_url: str,
    prompt: str,
    max_tokens: int,
    request_id: int,
) -> Dict[str, Any]:
    """Sends a single non-streaming generation request and records metrics."""
    url = f"{base_url.rstrip('/')}/api/v1/generate"
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    t0 = time.perf_counter()
    response = await client.post(url, json=payload, timeout=60.0)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    if response.status_code != 200:
        return {
            "success": False,
            "status_code": response.status_code,
            "error": response.text,
            "request_id": request_id,
        }

    data = response.json()
    metrics = data.get("metrics", {})
    output_tokens = len(data.get("output_tokens", []))

    return {
        "success": True,
        "request_id": request_id,
        "total_latency_ms": latency_ms,
        "output_tokens": output_tokens,
        "ttft_ms": metrics.get("ttft_ms", latency_ms),
        "avg_itl_ms": metrics.get("avg_itl_ms", 0.0),
        "finish_reason": data.get("finish_reason", "unknown"),
    }


async def send_stream_request(
    client: httpx.AsyncClient,
    base_url: str,
    prompt: str,
    max_tokens: int,
    request_id: int,
) -> Dict[str, Any]:
    """Sends an SSE streaming request and measures token arrival intervals."""
    url = f"{base_url.rstrip('/')}/api/v1/stream"
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    t0 = time.perf_counter()
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    itl_list: List[float] = []
    token_count = 0
    final_metrics: Dict[str, Any] = {}

    try:
        async with client.stream("POST", url, json=payload, timeout=60.0) as response:
            if response.status_code != 200:
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "error": await response.aread(),
                    "request_id": request_id,
                }

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                chunk_str = line[6:].strip()
                if not chunk_str:
                    continue

                now = time.perf_counter()
                chunk_data = json.loads(chunk_str)

                if "token" in chunk_data and chunk_data["token"]:
                    token_count += 1
                    if first_token_time is None:
                        first_token_time = now
                    else:
                        itl_list.append((now - last_token_time) * 1000.0)
                    last_token_time = now

                if chunk_data.get("metrics"):
                    final_metrics = chunk_data["metrics"]

        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        ttft_ms = (first_token_time - t0) * 1000.0 if first_token_time else total_latency_ms
        avg_itl_ms = statistics.mean(itl_list) if itl_list else 0.0

        return {
            "success": True,
            "request_id": request_id,
            "total_latency_ms": total_latency_ms,
            "output_tokens": token_count,
            "ttft_ms": ttft_ms,
            "avg_itl_ms": avg_itl_ms,
            "final_metrics": final_metrics,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "request_id": request_id,
        }


def compute_percentiles(values: List[float]) -> Dict[str, float]:
    """Calculates min, mean, p50, p90, p99, and max from a list of floats."""
    if not values:
        return {"min": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}

    s = sorted(values)
    n = len(s)
    return {
        "min": round(s[0], 2),
        "mean": round(statistics.mean(s), 2),
        "p50": round(s[int(n * 0.50)], 2),
        "p90": round(s[min(int(n * 0.90), n - 1)], 2),
        "p99": round(s[min(int(n * 0.99), n - 1)], 2),
        "max": round(s[-1], 2),
    }


async def run_benchmark(
    base_url: str,
    concurrency: int,
    total_requests: int,
    max_tokens: int,
    stream: bool,
    in_process: bool = False,
) -> None:
    """Executes concurrent load test and prints performance report."""
    print("=" * 70)
    print("               BATCHSERVE INFERENCE BENCHMARK")
    print("=" * 70)
    print(f"Target URL:         {base_url}")
    print(f"Mode:               {'In-Process (ASGITransport)' if in_process else 'HTTP Client'}")
    print(f"Protocol:           {'Server-Sent Events (SSE)' if stream else 'REST Batch /generate'}")
    print(f"Concurrent Clients: {concurrency}")
    print(f"Total Requests:     {total_requests}")
    print(f"Max New Tokens:     {max_tokens}")
    print("=" * 70)

    # Client setup
    if in_process:
        from app.main import app
        from app.services.engine import engine
        await engine.start()
        base_url = "http://testserver"
        transport = httpx.ASGITransport(app=app)
        client = httpx.AsyncClient(transport=transport, base_url=base_url)
    else:
        client = httpx.AsyncClient()

    # Semaphore for concurrency limiting
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(req_id: int) -> Dict[str, Any]:
        prompt = PROMPT_BANK[req_id % len(PROMPT_BANK)]
        async with semaphore:
            if stream:
                return await send_stream_request(client, base_url, prompt, max_tokens, req_id)
            else:
                return await send_generate_request(client, base_url, prompt, max_tokens, req_id)

    print(f"[*] Dispatching {total_requests} requests across {concurrency} concurrent workers...")
    bench_start = time.perf_counter()
    tasks = [worker(i) for i in range(total_requests)]
    results = await asyncio.gather(*tasks)
    bench_duration = time.perf_counter() - bench_start

    # Fetch server telemetry
    metrics_data = {}
    try:
        metrics_resp = await client.get(f"{base_url.rstrip('/')}/metrics")
        if metrics_resp.status_code == 200:
            metrics_data = metrics_resp.json()
    except Exception as e:
        print(f"[!] Warning: Unable to fetch /metrics: {e}")

    await client.aclose()

    # Aggregate statistics
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    total_tokens = sum(r.get("output_tokens", 0) for r in successful)
    ttft_values = [r["ttft_ms"] for r in successful if "ttft_ms" in r]
    itl_values = [r["avg_itl_ms"] for r in successful if "avg_itl_ms" in r and r["avg_itl_ms"] > 0]
    latencies = [r["total_latency_ms"] for r in successful if "total_latency_ms" in r]

    ttft_stats = compute_percentiles(ttft_values)
    itl_stats = compute_percentiles(itl_values)
    latency_stats = compute_percentiles(latencies)

    rps = len(successful) / bench_duration if bench_duration > 0 else 0.0
    throughput = total_tokens / bench_duration if bench_duration > 0 else 0.0

    print("\n" + "=" * 70)
    print("                    BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Time:             {bench_duration:.2f} s")
    print(f"Requests Completed:     {len(successful)} / {total_requests} ({len(successful)/total_requests*100:.1f}%)")
    print(f"Requests Failed:        {len(failed)}")
    print(f"Total Output Tokens:    {total_tokens}")
    print(f"Request Throughput:     {rps:.2f} req/s")
    print(f"Token Throughput:       {throughput:.2f} tokens/s")
    print("-" * 70)
    print("Latency Distribution (ms):")
    print(f"  Total Latency:   p50={latency_stats['p50']}ms | p90={latency_stats['p90']}ms | p99={latency_stats['p99']}ms | max={latency_stats['max']}ms")
    print(f"  Time-To-First-Tok: p50={ttft_stats['p50']}ms | p90={ttft_stats['p90']}ms | p99={ttft_stats['p99']}ms | max={ttft_stats['max']}ms")
    print(f"  Inter-Token Lat:   p50={itl_stats['p50']}ms | p90={itl_stats['p90']}ms | p99={itl_stats['p99']}ms | max={itl_stats['max']}ms")
    print("-" * 70)
    if metrics_data:
        print("Paged KV-Cache Memory & Scheduler Status:")
        print(f"  Total Physical Blocks:     {metrics_data.get('total_blocks')}")
        print(f"  Used Blocks:               {metrics_data.get('used_blocks')}")
        print(f"  Free Blocks:               {metrics_data.get('free_blocks')}")
        print(f"  KV-Cache Utilization:      {metrics_data.get('kv_cache_utilization_pct')}%")
        print(f"  Prefix Cache Hit Rate:     {metrics_data.get('cache_hit_rate', 0.0) * 100:.2f}%")
        print(f"  Preemption Events:         {metrics_data.get('preempted_sequences')}")
        print(f"  Completed Sequences Total: {metrics_data.get('total_completed_sequences')}")
    print("=" * 70)


def main() -> None:
    """CLI entrypoint for benchmark."""
    parser = argparse.ArgumentParser(description="BatchServe LLM Inference Benchmark")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="Base URL of BatchServe")
    parser.add_argument("--concurrency", "-c", type=int, default=8, help="Number of concurrent client workers")
    parser.add_argument("--total-requests", "-n", type=int, default=32, help="Total number of requests")
    parser.add_argument("--max-tokens", "-m", type=int, default=24, help="Max tokens to generate per request")
    parser.add_argument("--stream", action="store_true", help="Benchmark SSE streaming endpoint")
    parser.add_argument("--in-process", action="store_true", help="Run benchmark in-process via ASGITransport")

    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            base_url=args.url,
            concurrency=args.concurrency,
            total_requests=args.total_requests,
            max_tokens=args.max_tokens,
            stream=args.stream,
            in_process=args.in_process,
        )
    )


if __name__ == "__main__":
    main()
