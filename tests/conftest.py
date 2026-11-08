"""Pytest fixtures for BatchServe test suite."""

import asyncio
from typing import AsyncGenerator
import pytest
import pytest_asyncio
import httpx

from app.main import create_app
from app.models.sequence import Sequence
from app.services.block_manager import BlockManager
from app.services.engine import InferenceEngine
from app.services.scheduler import Scheduler
from app.services.tokenizer import SimpleTokenizer


@pytest.fixture
def tokenizer() -> SimpleTokenizer:
    """Provides a fresh tokenizer instance."""
    return SimpleTokenizer(vocab_size=500)


@pytest.fixture
def block_manager() -> BlockManager:
    """Provides a dedicated BlockManager instance with small block size for testing."""
    return BlockManager(total_blocks=16, block_size=4, enable_prefix_caching=False)


@pytest.fixture
def prefix_block_manager() -> BlockManager:
    """Provides a BlockManager with prefix caching enabled."""
    return BlockManager(total_blocks=16, block_size=4, enable_prefix_caching=True)


@pytest.fixture
def scheduler(block_manager: BlockManager) -> Scheduler:
    """Provides a scheduler wired to the test block manager."""
    return Scheduler(block_manager=block_manager, max_batch_size=4, watermark_blocks=1)


@pytest.fixture
def engine(block_manager: BlockManager, scheduler: Scheduler, tokenizer: SimpleTokenizer) -> InferenceEngine:
    """Provides an isolated InferenceEngine instance."""
    return InferenceEngine(
        block_manager=block_manager,
        scheduler=scheduler,
        tokenizer=tokenizer,
    )


def create_mock_sequence(
    seq_id: str,
    prompt_tokens: list[int],
    max_new_tokens: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop_sequences: list[str] = None,
) -> Sequence:
    """Helper creating a test sequence with specified token list."""
    return Sequence(
        seq_id=seq_id,
        prompt="test prompt",
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_sequences=stop_sequences or [],
    )


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Provides an in-process async test client for FastAPI routes."""
    from app.services.engine import engine as global_engine

    app = create_app()
    await global_engine.start()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    await global_engine.stop()

