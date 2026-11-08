"""BatchServe Engine Configuration.

Defines all runtime settings, Paged KV-cache parameters, continuous
batching scheduler thresholds, and model simulator constants.
"""

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global system configuration backed by environment variables."""

    # Server Configuration
    PROJECT_NAME: str = "BatchServe"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_V1_STR: str = "/api/v1"
    LOG_LEVEL: str = "INFO"

    # Paged KV-Cache Configuration
    BLOCK_SIZE: int = 16  # Number of tokens per physical memory block
    TOTAL_BLOCKS: int = 128  # Total physical blocks (128 * 16 = 2,048 tokens capacity)
    TOTAL_SWAP_BLOCKS: int = 64  # Secondary CPU swap memory blocks
    WATERMARK_BLOCKS: int = 4  # Headroom reserved before triggering preemption
    ENABLE_PREFIX_CACHING: bool = True  # Cache shared prompt prefixes across requests
    ENABLE_SWAPPING: bool = True  # Support memory swapping to CPU block pool

    # Continuous Batch Scheduler Configuration
    MAX_BATCH_SIZE: int = 32  # Maximum number of concurrently running sequences
    MAX_WAITING_QUEUE_SIZE: int = 256  # Backpressure ceiling for pending requests
    CHUNKED_PREFILL_SIZE: int = 32  # Maximum prompt tokens processed per iteration chunk
    ENABLE_CHUNKED_PREFILL: bool = True  # Piggyback prefill chunks onto decode iterations
    SCHEDULING_POLICY: Literal["FCFS", "PRIORITY"] = "FCFS"
    PREEMPTION_MODE: Literal["recompute", "swap"] = "recompute"

    # Lightweight Model Simulator Configuration
    VOCAB_SIZE: int = 1000  # Lightweight vocabulary size
    MAX_MODEL_LEN: int = 2048  # Context length ceiling (prompt + generation)
    SIMULATED_STEP_LATENCY_MS: float = 4.0  # Simulated micro-step compute duration in ms
    ENABLE_SIMULATED_LATENCY: bool = True

    model_config = SettingsConfigDict(
        env_prefix="BATCHSERVE_",
        env_file=".env",
        extra="ignore",
    )


# Global settings singleton
settings = Settings()

