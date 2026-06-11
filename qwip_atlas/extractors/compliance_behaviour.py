from __future__ import annotations

from collections import defaultdict
from typing import Any

from qwip_atlas.config import ComplianceBehaviourRunConfig
from qwip_atlas.io import iter_jsonl, write_json
from qwip_atlas.layers import inspect_layer, layers_container, resolve_layers


def _load_model_and_tokenizer(cfg: ComplianceBehaviourRunConfig, hf_token: str | None):
    from qwip_atlas.extractors.local_census import _load_model_and_tokenizer as _load

    # AtlasRunConfig and ComplianceBehaviourRunConfig share the `model` fields used by _load.
    return _load(cfg, hf_token)  # type: ignore[arg-type]


def _register_hooks(per_layer_info: dict[int, dict], captured: dict[tuple[int, str], Any]):
    from qwip_atlas.extractors.local_census import _register_hooks

    return _register_hooks(per_layer_info, captured)


def _load_prompts(corpus, label: str) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(corpus.path))
    if not rows:
        raise ValueError(f"{label} corpus is empty: {corpus.path}")
    missing = [i for i, row in enumerate(rows) if corpus.prompt_key not in row]
    if missing:
        raise ValueError(f"{label} corpus missing {corpus.prompt_key!r}: first bad row {missing[0]}")
    return rows


def _check_per_head(tensor, head_dim: int | None):
    if tensor is None or not head_dim or tensor.shape[-1] % head_dim != 0:
        return None
    return tensor


def _last_token_components(captured: dict[tuple[int, str], Any], layer: int, info: dict, batch_idx: int, seq_len: int):
    import torch

    act_fn = info["mlp"]["act_fn"] or torch.nn.functional.silu
    head_dim = info["attn"]["head_dim"]
    sl = slice(-seq_len, None)

    mlp_hidden = captured.get((layer, "mlp_hidden"))
    if mlp_hidden is None:
        return {}

    # ⚡ Bolt Optimization:
    # Previously, act_fn was applied to the entire batch and sequence length,
    # and _safe_per_head triggered expensive reshapes across the full batch.
    # We now slice the raw tensor first, avoiding O(batch_size * seq_len * d_model)
    # unnecessary computation, then apply operations specifically on the needed slice.
    raw_tensors = {
        "mlp": mlp_hidden,
        "gate": captured.get((layer, "gate_pre")),
        "up": captured.get((layer, "up")),
        "attn": captured.get((layer, "attn_out")),
        "heads": _check_per_head(captured.get((layer, "attn_pre")), head_dim),
        "q": _check_per_head(captured.get((layer, "q")), head_dim),
        "k": _check_per_head(captured.get((layer, "k")), head_dim),
        "v": _check_per_head(captured.get((layer, "v")), head_dim),
    }

    out = {}
    for name, tensor in raw_tensors.items():
        if tensor is None:
            continue

        # Slice the tensor to get the last token first
        t = tensor[batch_idx, sl][-1]

        if name == "gate":
            t = act_fn(t)

        out[name] = t.reshape(-1).numpy()
    return out


def _binary_fstat(pos, neg):
    import numpy as np

    pos = np.asarray(pos, dtype=np.float32)
    neg = np.asarray(neg, dtype=np.float32)
    n_pos, n_neg = pos.shape[0], neg.shape[0]
    mean_pos = pos.mean(axis=0)
    mean_neg = neg.mean(axis=0)
    std_pos = pos.std(axis=0)
    std_neg = neg.std(axis=0)
    grand = (mean_pos * n_pos + mean_neg * n_neg) / max(n_pos + n_neg, 1)
    ss_between = n_pos * (mean_pos - grand) ** 2 + n_neg * (mean_neg - grand) ** 2
    ss_within = ((pos - mean_pos) ** 2).sum(axis=0) + ((neg - mean_neg) ** 2).sum(axis=0)
    df_within = max(n_pos + n_neg - 2, 1)
    fstat = ss_between / (ss_within / df_within + 1e-10)
    return {
        "fstat": fstat.astype(np.float32),
        "delta": (mean_pos - mean_neg).astype(np.float32),
        "mean_pos": mean_pos.astype(np.float32),
        "mean_neg": mean_neg.astype(np.float32),
        "std_pos": std_pos.astype(np.float32),
        "std_neg": std_neg.astype(np.float32),
    }


def run_compliance_behaviour(cfg: ComplianceBehaviourRunConfig, hf_token: str | None = None) -> dict:
    """Compute binary behavior-axis feature scores.

    Output intentionally keeps `mean_corp` / `mean_auth` aliases for compatibility
    with the existing atlas merge code. Use labels in metadata for generic runs.
    """
    import numpy as np
    import torch
    from tqdm import tqdm

    positive = _load_prompts(cfg.positive_corpus, cfg.positive_label)
    negative = _load_prompts(cfg.negative_corpus, cfg.negative_label)
    corpus = [(row, 1) for row in positive] + [(row, 0) for row in negative]
    print(f"[compliance_behaviour] {len(positive)} {cfg.positive_label} + {len(negative)} {cfg.negative_label}")

    model, tokenizer = _load_model_and_tokenizer(cfg, hf_token)
    layers = resolve_layers(model)
    text_cfg = getattr(model.config, "text_config", model.config)

    per_layer_info: dict[int, dict] = {}
    for layer in cfg.layers:
        if layer >= len(layers):
            print(f"[compliance_behaviour] [warn] layer {layer} out of range; model has {len(layers)} layers")
            continue
        info = inspect_layer(layers[layer], text_cfg)
        if info["mlp"]["down_proj"] is None:
            print(f"[compliance_behaviour] [warn] layer {layer} has no MLP down projection; skipping")
            continue
        per_layer_info[layer] = info
        mlp, attn = info["mlp"], info["attn"]
        print(
            f"[compliance_behaviour] layer {layer:>2}: d_mlp={mlp['d_mlp']} "
            f"Q={attn['n_heads']}x{attn['head_dim']} KV={attn['n_kv_heads']}x{attn['head_dim']}"
        )

    if not per_layer_info:
        raise RuntimeError("No valid target layers")

    if cfg.truncate_to_deepest_layer:
        deepest = max(per_layer_info)
        if deepest + 1 < len(layers):
            container = layers_container(model)
            container.layers = container.layers[: deepest + 1]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[compliance_behaviour] truncated forward graph to {deepest + 1} layers")

    values: dict[int, dict[str, dict[int, list]]] = {
        layer: defaultdict(lambda: {1: [], 0: []}) for layer in per_layer_info
    }

    for start in tqdm(range(0, len(corpus), cfg.batch_size), desc="batches", unit="batch"):
        batch = corpus[start:start + cfg.batch_size]
        prompts = [
            row[cfg.positive_corpus.prompt_key] if label == 1 else row[cfg.negative_corpus.prompt_key]
            for row, label in batch
        ]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.model.max_length,
        )
        seq_lens = enc["attention_mask"].sum(dim=1).tolist()
        device = getattr(model, "device", None) or next(model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}

        captured: dict[tuple[int, str], Any] = {}
        handles = _register_hooks(per_layer_info, captured)
        with torch.no_grad():
            model(**enc, use_cache=False)
        for handle in handles:
            handle.remove()

        for layer, info in per_layer_info.items():
            for batch_idx, ((_, label), seq_len) in enumerate(zip(batch, seq_lens)):
                comps = _last_token_components(captured, layer, info, batch_idx, int(seq_len))
                for comp, vector in comps.items():
                    if comp in cfg.components:
                        values[layer][comp][label].append(vector)
        captured.clear()

    result: dict[str, dict] = {}
    for layer in sorted(values):
        layer_out = {}
        for comp, groups in sorted(values[layer].items()):
            if not groups[1] or not groups[0]:
                continue
            stats = _binary_fstat(groups[1], groups[0])
            head_dim = per_layer_info[layer]["attn"]["head_dim"] if comp in {"heads", "q", "k", "v"} else None
            is_per_head = bool(head_dim and stats["fstat"].shape[0] % int(head_dim) == 0)
            layer_out[comp] = {
                "fstat": stats["fstat"].tolist(),
                "delta": stats["delta"].tolist(),
                "mean_corp": stats["mean_pos"].tolist(),
                "mean_auth": stats["mean_neg"].tolist(),
                "std_corp": stats["std_pos"].tolist(),
                "std_auth": stats["std_neg"].tolist(),
                "mean_positive": stats["mean_pos"].tolist(),
                "mean_negative": stats["mean_neg"].tolist(),
                "std_positive": stats["std_pos"].tolist(),
                "std_negative": stats["std_neg"].tolist(),
                "n_corporate": len(groups[1]),
                "n_authentic": len(groups[0]),
                "n_positive": len(groups[1]),
                "n_negative": len(groups[0]),
                "positive_label": cfg.positive_label,
                "negative_label": cfg.negative_label,
                "n_features": int(stats["fstat"].shape[0]),
                "is_per_head": is_per_head,
                "head_dim": int(head_dim) if head_dim else None,
            }
            top = np.argsort(stats["fstat"])[-3:][::-1]
            print(
                f"[compliance_behaviour] L{layer:02d} {comp:<5} top="
                + ", ".join(f"{int(i)} F={stats['fstat'][i]:.1f} d={stats['delta'][i]:.3f}" for i in top)
            )
        result[str(layer)] = layer_out

    write_json(cfg.output, result)
    print(f"[compliance_behaviour] wrote {cfg.output}")
    return result
