"""Contract and integration tests for BatchServe API routes."""

import asyncio
import json
import pytest
import httpx


@pytest.mark.asyncio
async def test_health_endpoint(async_client: httpx.AsyncClient) -> None:
    """Tests GET /health returns operational status."""
    response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["engine"] == "BatchServe"
    assert "total_blocks" in data
    assert "free_blocks" in data


@pytest.mark.asyncio
async def test_metrics_endpoint(async_client: httpx.AsyncClient) -> None:
    """Tests GET /metrics returns full telemetry."""
    response = await async_client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "cache_hit_rate" in data
    assert "kv_cache_utilization_pct" in data
    assert "tokens_per_second" in data
    assert "avg_ttft_ms" in data


@pytest.mark.asyncio
async def test_generate_endpoint_contract(async_client: httpx.AsyncClient) -> None:
    """Tests POST /api/v1/generate returns complete response with metrics breakdown."""
    payload = {
        "prompt": "Explain Paged KV-Cache architecture in LLM serving.",
        "max_tokens": 12,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    response = await async_client.post("/api/v1/generate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "id" in data
    assert data["prompt"] == payload["prompt"]
    assert isinstance(data["text"], str)
    assert len(data["output_tokens"]) <= 12
    assert data["finish_reason"] in ("length", "stop")

    metrics = data["metrics"]
    assert "ttft_ms" in metrics
    assert "avg_itl_ms" in metrics
    assert "total_latency_ms" in metrics
    assert "tokens_per_second" in metrics


@pytest.mark.asyncio
async def test_stream_sse_endpoint_contract(async_client: httpx.AsyncClient) -> None:
    """Tests POST /api/v1/stream Server-Sent Events contract."""
    payload = {
        "prompt": "Stream tokens one by one.",
        "max_tokens": 8,
        "temperature": 0.5,
        "top_p": 0.9,
    }

    chunks = []
    async with async_client.stream("POST", "/api/v1/stream", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        async for line in response.aiter_lines():
            if line.startswith("data: "):
                raw_json = line[6:].strip()
                if raw_json:
                    chunk_obj = json.loads(raw_json)
                    chunks.append(chunk_obj)

    assert len(chunks) >= 1
    # Verify sequential step ordering
    steps = [c["step"] for c in chunks]
    assert steps == list(range(1, len(chunks) + 1))

    # Final chunk must specify finish_reason and metrics
    final_chunk = chunks[-1]
    assert final_chunk["finish_reason"] in ("length", "stop")
    assert final_chunk["metrics"] is not None
    assert "ttft_ms" in final_chunk["metrics"]


@pytest.mark.asyncio
async def test_validation_errors(async_client: httpx.AsyncClient) -> None:
    """Tests Pydantic input validations for invalid inputs."""
    # Empty prompt
    resp1 = await async_client.post("/api/v1/generate", json={"prompt": ""})
    assert resp1.status_code == 422

    # Negative max_tokens
    resp2 = await async_client.post("/api/v1/generate", json={"prompt": "Valid", "max_tokens": 0})
    assert resp2.status_code == 422

    # Out-of-bounds temperature
    resp3 = await async_client.post("/api/v1/generate", json={"prompt": "Valid", "temperature": 3.5})
    assert resp3.status_code == 422


@pytest.mark.asyncio
async def test_concurrent_api_requests(async_client: httpx.AsyncClient) -> None:
    """Verifies that multiple concurrent requests are processed without deadlocks or memory leaks."""
    payloads = [
        {"prompt": f"Concurrent sequence {i}", "max_tokens": 6}
        for i in range(8)
    ]

    tasks = [async_client.post("/api/v1/generate", json=p) for p in payloads]
    responses = await asyncio.gather(*tasks)

    for resp in responses:
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["output_tokens"]) > 0

    # Verify memory deallocation after all requests finish
    metrics_resp = await async_client.get("/metrics")
    assert metrics_resp.status_code == 200
    metrics_data = metrics_resp.json()
    assert metrics_data["active_sequences"] == 0

