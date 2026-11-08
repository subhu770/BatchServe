<div align="center">

# ⚡ BatchServe

### High-Throughput LLM Serving Engine with Paged KV-Cache & Continuous Batching

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Paged KV-Cache](https://img.shields.io/badge/Memory-Paged_KV--Cache-orange.svg)](https://arxiv.org/abs/2309.06180)
[![Continuous Batching](https://img.shields.io/badge/Scheduling-Continuous_Batching-purple.svg)](https://www.usenix.org/conference/osdi22/presentation/yu)
[![SSE Streaming](https://img.shields.io/badge/Streaming-Server--Sent_Events-green.svg)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
[![Pytest](https://img.shields.io/badge/tests-20%20passed-brightgreen.svg?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Docker](https://img.shields.io/badge/Docker-Multi--stage-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <b>Eliminating external memory fragmentation and inference pipeline bubbles through iteration-level continuous scheduling and virtual memory paged KV-cache allocation.</b>
</p>

---

</div>

## 📌 1. Problem Statement: Naive Static Serving vs. BatchServe

Traditional LLM inference runtimes suffer from two fundamental systems-level bottlenecks: **memory fragmentation** and **compute bubbles**.

### The Naive Batching Bottleneck
1. **Static Batch Padding & Compute Inefficiency**:
   - In naive static batching, $N$ concurrent requests are grouped together. Sequences have variable prompt and output lengths ($L_1, L_2, \dots, L_N$).
   - Static batches must be zero-padded to the maximum length $\max_i(L_i)$. For a batch where one sequence generates 500 tokens and others generate 30 tokens, 94% of compute cycles on shorter sequences are wasted computing dummy padding tokens.
   - **Head-of-Line Blocking**: Early-finishing requests cannot leave the GPU/CPU engine until the *longest* sequence finishes, introducing latency spikes and poor GPU utilization.

2. **Severe Memory Fragmentation**:
   - Traditional engines allocate a contiguous tensor for each sequence sized to `max_model_len` (e.g., 2,048 or 4,096 tokens) in anticipation of worst-case generation.
   - Because generation is auto-regressive, most of this reserved memory sits unutilized.
   - **External Fragmentation**: Dynamic sequence allocations and deallocations leave non-contiguous memory holes that cannot satisfy incoming allocations even when aggregate free memory is abundant. Up to **60% to 80%** of device memory is wasted.

```
┌────────────────────────────────────────────────────────────────────────┐
│ NAIVE STATIC BATCHING (Memory Waste & Padding Bubbles)                 │
├────────────────────────────────────────────────────────────────────────┤
│ Seq 1: [Tok 1][Tok 2][Tok 3] [ PAD ][ PAD ][ PAD ][ PAD ] (Wasted FLOPs)│
│ Seq 2: [Tok 1][Tok 2][Tok 3][Tok 4][Tok 5][Tok 6][Tok 7] (Longest)    │
│ Seq 3: [Tok 1] [ PAD ][ PAD ][ PAD ][ PAD ][ PAD ][ PAD ] (Trapped)   │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ BATCHSERVE CONTINUOUS BATCHING (Zero Waste, Iteration-Level Dynamic)   │
├────────────────────────────────────────────────────────────────────────┤
│ Step t:   [Seq 1: Tok 3] [Seq 2: Tok 6] [Seq 4: Tok 1 (Newly Admitted)]│
│ Step t+1: [Seq 1: FINISHED ➔ Freed] [Seq 2: Tok 7] [Seq 5: Admitted]   │
└────────────────────────────────────────────────────────────────────────┘
```

### The BatchServe Solution
- **Iteration-Level Continuous Batching**: Scheduling decisions happen at the individual token generation step. As soon as any sequence completes, it is evicted immediately, its memory is freed, and a waiting sequence is admitted without waiting for the batch to complete.
- **Paged KV-Cache Memory Management**: Inspired by operating system virtual memory paging and vLLM's PagedAttention, token memory is segmented into fixed-size physical blocks ($B = 16$ tokens). Logical token positions are translated to physical block tables on-the-fly, completely eliminating external memory fragmentation.

---

## 🏛️ 2. Core Architecture & Data Pipeline

```
                               ┌──────────────────────────────────┐
                               │   Client Generation Requests     │
                               │  (REST /generate & SSE /stream)  │
                               └────────────────┬─────────────────┘
                                                │
                                                ▼
                               ┌──────────────────────────────────┐
                               │       FastAPI Application        │
                               │      [app/api/routes.py]         │
                               └────────────────┬─────────────────┘
                                                │
                         ┌──────────────────────┴──────────────────────┐
                         ▼                                             ▼
          ┌─────────────────────────────┐               ┌─────────────────────────────┐
          │     Inference Engine        │               │   Paged KV-Cache Allocator  │
          │    [services/engine.py]     │◄─────────────►│ [services/block_manager.py] │
          └──────────────┬──────────────┘               └──────────────┬──────────────┘
                         │                                             │
                         ▼                                             ▼
          ┌─────────────────────────────┐               ┌─────────────────────────────┐
          │  Continuous Batch Scheduler │               │ Physical Block Pool (FIFO)  │
          │   [services/scheduler.py]   │               │ Logical-to-Physical Tables  │
          │ (WAITING/RUNNING/PREEMPTED) │               │ Radix/Hash Prefix Cache     │
          └─────────────────────────────┘               └─────────────────────────────┘
```

### A. Paged Block Manager ([`block_manager.py`](file:///c:/Users/Subham/ML/BatchServe/app/services/block_manager.py))
- **Physical Block Pool**: Maintains an explicit pool of $M = 128$ physical blocks, each containing $B = 16$ token slots (total capacity = 2,048 tokens).
- **Logical-to-Physical Mapping**: Maintains `block_tables: Dict[str, List[int]]` mapping each `seq_id` to its assigned non-contiguous physical block IDs.
- **Boundary Math**:
  - Prompt tokens require initially:
    $$\text{blocks\_needed} = \left\lceil \frac{N_{\text{prompt}}}{B} \right\rceil$$
  - At generation step $k$, a sequence with $T_k$ total tokens requests an additional block if and only if:
    $$T_k \pmod B = 0 \quad (T_k > 0)$$
  - Otherwise, the incoming token is stored within the remaining slots of the currently allocated physical block.
- **Deterministic Cleanup (Zero Leak)**:
  - Sequence termination calls `block_manager.free(seq)`.
  - Block reference counts are decremented; when `ref_count == 0`, blocks are recycled into the FIFO `free_blocks` queue, guaranteeing zero memory leaks.
- **Prefix Caching**: Identical prompt token blocks share physical memory blocks across different requests via reference counting.

```
Sequence Logical Tokens: [ 0 ... 15 ] [ 16 ... 31 ] [ 32 ... 47 ]
Logical Blocks:               Block 0        Block 1        Block 2
                                 │              │              │
Logical-to-Physical Table:       ▼              ▼              ▼
Physical Memory Blocks:      [ Phys #7 ]    [ Phys #14 ]   [ Phys #2 ]
```

### B. Continuous Iteration Scheduler ([`scheduler.py`](file:///c:/Users/Subham/ML/BatchServe/app/services/scheduler.py))
The continuous batch scheduler manages sequence transitions through a dynamic state machine:
- **`WAITING`**: Sequence received, queued awaiting memory admission.
- **`RUNNING`**: Active in the continuous batch. Generates one token per iteration step.
- **`PREEMPTED`**: Memory pressure hit watermark threshold; blocks safely freed, sequence returned to waiting queue.
- **`FINISHED`**: Hit `<|eos|>`, reached `max_new_tokens`, or encountered stop string.

```
                   ┌──────────────┐
                   │   Incoming   │
                   └──────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │    WAITING    │◄──────────────┐
                  └───────┬───────┘               │
      Memory Available    │                       │ Preemption
      (Headroom Checked)  ▼                       │ (Watermark breached)
                  ┌───────────────┐               │
                  │    RUNNING    ├───────────────┘
                  └───────┬───────┘
                          │ Finished (EOS / Max Tokens / Stop String)
                          ▼
                  ┌───────────────┐
                  │   FINISHED    │ ➔ Deallocate blocks to free pool
                  └───────────────┘
```

#### Scheduler Iteration Step Loop:
1. **Eviction**: Iterates over running sequences. Evicts finished sequences and frees their KV blocks immediately.
2. **Headroom & Preemption Protection**:
   - Calculates the number of running sequences crossing block boundaries:
     $$\text{blocks\_needed} = \sum_{s \in \text{Running}} \mathbb{I}(T_s \pmod B = 0)$$
   - If available headroom breaches safety threshold:
     $$\text{num\_free\_blocks} - \text{blocks\_needed} < \text{watermark\_blocks}$$
     the scheduler preempts victim sequences (LIFO / lowest priority) back to `WAITING` and deallocates their blocks to prevent Out-Of-Memory (OOM) crashes.
3. **Admission**: Admits pending sequences from `waiting_queue` into the running batch up to `max_batch_size`. Implements anti-thrashing by barring sequences preempted in the current tick from immediate re-admission.

### C. Async Inference Engine & Streaming ([`engine.py`](file:///c:/Users/Subham/ML/BatchServe/app/services/engine.py))
- **Logits & Sampling**: Applies temperature scaling ($z_i / T$) and nucleus (top-p) sampling over vocabulary probabilities.
- **Async Step Loop**: Runs as a non-blocking background task yielding control (`await asyncio.sleep(0)`) to maintain concurrent network I/O.
- **Streaming Pipeline**: Pushes each generated token to the sequence's async `token_queue`, immediately yielding Server-Sent Events (SSE) to connected HTTP clients.

---

## 📊 3. Benchmark & Verification Telemetry

### A. Pytest Verification Suite (20 Passing Tests)
The comprehensive test suite in [`tests/`](file:///c:/Users/Subham/ML/BatchServe/tests/) verifies all architectural invariants:

```bash
python -m pytest -v
```

```text
============================= test session starts =============================
platform win32 -- Python 3.14.5, pytest-9.1.1, pluggy-1.6.0
collected 20 items

tests/test_api.py::test_health_endpoint PASSED                           [  5%]
tests/test_api.py::test_metrics_endpoint PASSED                          [ 10%]
tests/test_api.py::test_generate_endpoint_contract PASSED                [ 15%]
tests/test_api.py::test_stream_sse_endpoint_contract PASSED              [ 20%]
tests/test_api.py::test_validation_errors PASSED                         [ 25%]
tests/test_api.py::test_concurrent_api_requests PASSED                   [ 30%]
tests/test_block_manager.py::test_pool_initialization PASSED             [ 35%]
tests/test_block_manager.py::test_prompt_block_allocation PASSED         [ 40%]
tests/test_block_manager.py::test_append_token_boundary_math PASSED      [ 45%]
tests/test_block_manager.py::test_zero_kv_cache_leaks_on_deallocation PASSED [ 50%]
tests/test_block_manager.py::test_out_of_memory_handling PASSED          [ 55%]
tests/test_block_manager.py::test_prefix_caching_sharing PASSED          [ 60%]
tests/test_engine.py::test_engine_single_step PASSED                     [ 65%]
tests/test_engine.py::test_engine_completion_lifecycle PASSED            [ 70%]
tests/test_engine.py::test_engine_stop_sequence PASSED                   [ 75%]
tests/test_engine.py::test_engine_metrics_endpoint_data PASSED           [ 80%]
tests/test_scheduler.py::test_scheduler_admission_and_lifecycle PASSED   [ 85%]
tests/test_scheduler.py::test_continuous_batching_eviction PASSED        [ 90%]
tests/test_scheduler.py::test_max_batch_size_throttling PASSED           [ 95%]
tests/test_scheduler.py::test_preemption_under_kv_memory_pressure PASSED [100%]

============================= 20 passed in 1.13s ==============================
```

### B. High-Concurrency Benchmark Run ([`samples/benchmark.py`](file:///c:/Users/Subham/ML/BatchServe/samples/benchmark.py))

```bash
python samples/benchmark.py --in-process --concurrency 4 --total-requests 8 --max-tokens 16
```

```text
======================================================================
               BATCHSERVE INFERENCE BENCHMARK
======================================================================
Target URL:         http://127.0.0.1:8000
Mode:               In-Process (ASGITransport)
Protocol:           REST Batch /generate
Concurrent Clients: 4
Total Requests:     8
Max New Tokens:     16
======================================================================
[*] Dispatching 8 requests across 4 concurrent workers...

======================================================================
                    BENCHMARK RESULTS SUMMARY
======================================================================
Total Time:             0.91 s
Requests Completed:     8 / 8 (100.0%)
Requests Failed:        0
Total Output Tokens:    128
Request Throughput:     8.74 req/s
Token Throughput:       139.89 tokens/s
----------------------------------------------------------------------
Latency Distribution (ms):
  Total Latency:     p50=630.42ms | p90=642.54ms | p99=642.54ms | max=642.54ms
  Time-To-First-Tok: p50=12.29ms  | p90=16.19ms  | p99=16.19ms  | max=16.19ms
  Inter-Token Lat:   p50=40.51ms  | p90=40.51ms  | p99=40.51ms  | max=40.51ms
----------------------------------------------------------------------
Paged KV-Cache Memory & Scheduler Status:
  Total Physical Blocks:     128
  Used Blocks:               0
  Free Blocks:               128 (Zero Memory Leaks Verified)
  KV-Cache Utilization:      0.0%
  Prefix Cache Hit Rate:     0.00%
  Preemption Events:         0
  Completed Sequences Total: 8
======================================================================
```

---

## ⚡ 4. Quickstart Guide

### 1. Installation
```bash
git clone https://github.com/subhu770/BatchServe.git
cd BatchServe

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Server
```bash
# Standalone boot with Uvicorn
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or deploy with Docker:
```bash
docker-compose up --build -d
```

### 3. API Commands

#### A. Health Check
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "healthy",
  "engine": "BatchServe",
  "version": "1.0.0",
  "total_blocks": 128,
  "free_blocks": 128,
  "active_sequences": 0
}
```

#### B. Non-Streaming Generation (`POST /api/v1/generate`)
```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain Paged KV-Cache architecture in LLM serving.",
    "max_tokens": 24,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```
```json
{
  "id": "2787e67f-c1f9-467f-8be2-8924b1a45749",
  "prompt": "Explain Paged KV-Cache architecture in LLM serving.",
  "text": " Paged memory partitions continuous sequences into fixed physical blocks...",
  "output_tokens": [18, 45, 102, 33, 89],
  "finish_reason": "length",
  "metrics": {
    "arrival_time": 1729012345.12,
    "ttft_ms": 13.4,
    "avg_itl_ms": 4.1,
    "total_latency_ms": 112.5,
    "tokens_per_second": 213.33,
    "preemption_count": 0
  }
}
```

#### C. Server-Sent Events (SSE) Streaming (`POST /api/v1/stream`)
```bash
curl -N -X POST http://localhost:8000/api/v1/stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Stream generation tokens dynamically.",
    "max_tokens": 16,
    "temperature": 0.7
  }'
```
```text
data: {"id":"9c20...","token":" Stream","token_id":82,"step":1,"finish_reason":null,"metrics":null}

data: {"id":"9c20...","token":" generation","token_id":124,"step":2,"finish_reason":null,"metrics":null}

data: {"id":"9c20...","token":"","token_id":null,"step":16,"finish_reason":"length","metrics":{"ttft_ms":12.1,"avg_itl_ms":4.05,"total_latency_ms":68.2,"tokens_per_second":234.6}}
```

#### D. Metrics & Telemetry (`GET /metrics`)
```bash
curl http://localhost:8000/metrics
```
```json
{
  "timestamp": 1729012399.82,
  "cache_hit_rate": 0.15,
  "total_blocks": 128,
  "free_blocks": 120,
  "used_blocks": 8,
  "kv_cache_utilization_pct": 6.25,
  "active_sequences": 3,
  "waiting_sequences": 0,
  "preempted_sequences": 0,
  "total_completed_sequences": 24,
  "total_tokens_generated": 768,
  "avg_ttft_ms": 14.2,
  "avg_itl_ms": 4.05,
  "tokens_per_second": 238.5
}
```

#### E. Run Benchmark Suite
```bash
# In-process benchmark
python samples/benchmark.py --in-process --concurrency 8 --total-requests 32 --max-tokens 24

# SSE Streaming benchmark
python samples/benchmark.py --in-process --concurrency 8 --total-requests 32 --max-tokens 24 --stream

# Against live remote instance
python samples/benchmark.py --url http://localhost:8000 -c 16 -n 64 -m 32
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
