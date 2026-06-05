from __future__ import annotations

from collections import deque
from typing import Any


def parse_layer_spec(spec: str) -> list[int]:
    """Parse strings like `0,4,10-12` into sorted unique layer ids."""
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def layers_container(model: Any):
    """Find the module that owns the decoder `layers` ModuleList."""
    import torch.nn as nn

    queue = deque([model])
    while queue:
        module = queue.popleft()
        layers = getattr(module, "layers", None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return module
        for _, child in module.named_children():
            queue.append(child)
    raise RuntimeError(f"Cannot find decoder layers ModuleList on {type(model).__name__}")


def resolve_layers(model: Any) -> list[Any]:
    return list(layers_container(model).layers)


def inspect_layer(layer_mod: Any, text_cfg: Any) -> dict[str, Any]:
    """Resolve common MLP/attention projections without assuming a model family."""
    info: dict[str, Any] = {
        "mlp": {
            "module": None,
            "down_proj": None,
            "gate_proj": None,
            "up_proj": None,
            "act_fn": None,
            "d_mlp": None,
        },
        "attn": {
            "module": None,
            "q_proj": None,
            "k_proj": None,
            "v_proj": None,
            "o_proj": None,
            "n_heads": None,
            "n_kv_heads": None,
            "head_dim": None,
            "class_name": None,
        },
    }

    mlp = getattr(layer_mod, "mlp", None) or getattr(layer_mod, "feed_forward", None)
    if mlp is not None:
        info["mlp"]["module"] = mlp
        info["mlp"]["down_proj"] = getattr(mlp, "down_proj", None) or getattr(mlp, "wo", None)
        info["mlp"]["gate_proj"] = getattr(mlp, "gate_proj", None) or getattr(mlp, "w1", None)
        info["mlp"]["up_proj"] = getattr(mlp, "up_proj", None) or getattr(mlp, "w3", None)
        info["mlp"]["act_fn"] = getattr(mlp, "act_fn", None)
        down_proj = info["mlp"]["down_proj"]
        if down_proj is not None and hasattr(down_proj, "in_features"):
            info["mlp"]["d_mlp"] = down_proj.in_features

    attn = (
        getattr(layer_mod, "self_attn", None)
        or getattr(layer_mod, "attention", None)
        or getattr(layer_mod, "attn", None)
    )
    if attn is not None:
        info["attn"]["module"] = attn
        info["attn"]["class_name"] = type(attn).__name__
        info["attn"]["q_proj"] = getattr(attn, "q_proj", None)
        info["attn"]["k_proj"] = getattr(attn, "k_proj", None)
        info["attn"]["v_proj"] = getattr(attn, "v_proj", None)
        info["attn"]["o_proj"] = getattr(attn, "o_proj", None) or getattr(attn, "out_proj", None)

        head_dim = getattr(attn, "head_dim", None) or getattr(text_cfg, "head_dim", None)
        n_heads = getattr(attn, "num_heads", None) or getattr(text_cfg, "num_attention_heads", None)
        n_kv_heads = (
            getattr(attn, "num_key_value_heads", None)
            or getattr(text_cfg, "num_key_value_heads", None)
        )

        q_proj = info["attn"]["q_proj"]
        k_proj = info["attn"]["k_proj"]
        o_proj = info["attn"]["o_proj"]
        if head_dim is None and n_heads and o_proj is not None and hasattr(o_proj, "in_features"):
            head_dim = o_proj.in_features // n_heads
        if n_heads is None and head_dim and q_proj is not None and hasattr(q_proj, "out_features"):
            n_heads = q_proj.out_features // head_dim
        if n_kv_heads is None and head_dim and k_proj is not None and hasattr(k_proj, "out_features"):
            n_kv_heads = k_proj.out_features // head_dim
        if n_kv_heads is None:
            n_kv_heads = n_heads

        info["attn"]["n_heads"] = n_heads
        info["attn"]["n_kv_heads"] = n_kv_heads
        info["attn"]["head_dim"] = head_dim

    return info
