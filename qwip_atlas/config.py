from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    """Runtime model settings, deliberately outside the extractor code."""

    model_id: str
    revision: str | None = None
    trust_remote_code: bool = True
    dtype: str = "bfloat16"
    device_map: str | None = "auto"
    max_length: int = 512


@dataclass(frozen=True)
class CorpusSpec:
    """Local JSONL corpus settings."""

    path: Path
    prompt_key: str = "prompt"
    category_key: str = "category"
    bucket_key: str = "bucket"


@dataclass(frozen=True)
class AtlasRunConfig:
    """Config for a local activation-census run."""

    model: ModelSpec
    corpus: CorpusSpec
    layers: list[int]
    outdir: Path
    batch_size: int = 8
    components: set[str] = field(default_factory=lambda: {
        "mlp",
        "gate",
        "up",
        "attn",
        "heads",
        "q",
        "k",
        "v",
    })
    truncate_to_deepest_layer: bool = True
