from __future__ import annotations

import concurrent.futures
from typing import Any

from qwip_atlas.config import AtlasRunConfig
from qwip_atlas.io import iter_jsonl, write_json_array_stream
from qwip_atlas.layers import inspect_layer, layers_container, resolve_layers


def _torch_dtype(dtype_name: str):
    import torch

    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {dtype_name!r}; expected one of {sorted(aliases)}") from exc


def _bucket_for(row: dict[str, Any], category_key: str, bucket_key: str) -> str:
    if row.get(bucket_key):
        return str(row[bucket_key])
    category = str(row.get(category_key, "uncategorized"))
    return category.lower().replace(" & ", "_").replace(" ", "_")


def _load_model_and_tokenizer(cfg: AtlasRunConfig, hf_token: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_spec = cfg.model
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec.model_id,
        revision=model_spec.revision,
        trust_remote_code=model_spec.trust_remote_code,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    kwargs = {
        "revision": model_spec.revision,
        "trust_remote_code": model_spec.trust_remote_code,
        "token": hf_token,
        "torch_dtype": _torch_dtype(model_spec.dtype),
    }
    if model_spec.device_map:
        kwargs["device_map"] = model_spec.device_map

    model = AutoModelForCausalLM.from_pretrained(model_spec.model_id, **kwargs)
    model.eval()
    if not model_spec.device_map and torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer


def _register_hooks(per_layer_info: dict[int, dict], captured: dict[tuple[int, str], Any]):
    handles = []

    def make_hook(layer: int, key: str, take_input: bool = False):
        if take_input:
            def _hook(module, inputs):
                # Keep on GPU; bulk transfer after forward pass.
                captured[(layer, key)] = inputs[0].detach()
            return _hook

        def _hook(module, inputs, output):
            x = output[0] if isinstance(output, tuple) else output
            captured[(layer, key)] = x.detach()
        return _hook

    for layer, info in per_layer_info.items():
        mlp, attn = info["mlp"], info["attn"]
        if mlp["down_proj"] is not None:
            handles.append(mlp["down_proj"].register_forward_pre_hook(
                make_hook(layer, "mlp_hidden", take_input=True)
            ))
        if mlp["gate_proj"] is not None:
            handles.append(mlp["gate_proj"].register_forward_hook(make_hook(layer, "gate_pre")))
        if mlp["up_proj"] is not None:
            handles.append(mlp["up_proj"].register_forward_hook(make_hook(layer, "up")))
        if attn["q_proj"] is not None:
            handles.append(attn["q_proj"].register_forward_hook(make_hook(layer, "q")))
        if attn["k_proj"] is not None:
            handles.append(attn["k_proj"].register_forward_hook(make_hook(layer, "k")))
        if attn["v_proj"] is not None:
            handles.append(attn["v_proj"].register_forward_hook(make_hook(layer, "v")))
        if attn["o_proj"] is not None:
            handles.append(attn["o_proj"].register_forward_pre_hook(
                make_hook(layer, "attn_pre", take_input=True)
            ))
        if attn["module"] is not None:
            handles.append(attn["module"].register_forward_hook(make_hook(layer, "attn_out")))

    return handles


def _build_layer_rows(
    layer: int,
    info: dict,
    captured: dict[tuple[int, str], Any],
    rows: list[dict[str, Any]],
    seq_lens: list[int],
    cfg: AtlasRunConfig,
) -> list[dict[str, Any]]:
    import numpy as np
    import torch

    mlp_hidden = captured.get((layer, "mlp_hidden"))
    if mlp_hidden is None:
        return []

    act_fn = info["mlp"].get("act_fn") or torch.nn.functional.silu
    head_dim = info["attn"]["head_dim"]
    batch_size = len(rows)

    # Bring the whole batch to CPU once, as float32.
    # We do non-blocking copies via .to('cpu', non_blocking=True), then synchronize.
    use_cuda = mlp_hidden.device.type == "cuda"
    raw_tensors = {
        "gate": captured.get((layer, "gate_pre")),
        "up": captured.get((layer, "up")),
        "attn": captured.get((layer, "attn_out")),
        "attn_heads": captured.get((layer, "attn_pre")),
        "q_heads": captured.get((layer, "q")),
        "k_heads": captured.get((layer, "k")),
        "v_heads": captured.get((layer, "v")),
    }

    mh_cpu = mlp_hidden.float().to("cpu", non_blocking=use_cuda)
    cpu_tensors = {"mlp_hidden": mh_cpu}
    for key, t in raw_tensors.items():
        if t is None:
            continue
        cpu_tensors[key] = t.float().to("cpu", non_blocking=use_cuda)
    if use_cuda:
        torch.cuda.current_stream().synchronize()

    mh = mh_cpu.numpy()
    np_tensors = {k: v.numpy() for k, v in cpu_tensors.items() if k != "mlp_hidden"}

    # Apply the MLP activation function once to the whole batch slice.
    if "gate" in np_tensors:
        np_tensors["gate"] = act_fn(torch.from_numpy(np_tensors["gate"])).numpy()

    components = cfg.components
    key_to_component = {
        "gate": "gate",
        "up": "up",
        "attn": "attn",
        "attn_heads": "heads",
        "q_heads": "q",
        "k_heads": "k",
        "v_heads": "v",
    }

    out_rows = []
    for batch_index, (row, seq_len) in enumerate(zip(rows, seq_lens)):
        sl = slice(-int(seq_len), None)
        mh_slice = mh[batch_index, sl]

        out = {
            "id": row.get("id", f"p{row.get('_record_idx', batch_index):06d}"),
            "bucket": row.get("_bucket", "uncategorized"),
            "category": row.get(cfg.corpus.category_key, ""),
            "subcategory": row.get("subcategory", ""),
            "prompt": row[cfg.corpus.prompt_key],
            "is_contrast": row.get("is_contrast", False),
            "contrast_pair_id": row.get("contrast_pair_id"),
            "seq_len": int(seq_len),
            "max_token_idx": int(np.abs(mh_slice).sum(axis=-1).argmax().item()),
            "last_token": mh_slice[-1].tolist(),
            "mean_tokens": mh_slice.mean(0).tolist(),
        }

        for prefix, np_tensor in np_tensors.items():
            if key_to_component[prefix] not in components:
                continue
            t = np_tensor[batch_index, sl]
            if prefix.endswith("_heads"):
                if not head_dim or t.shape[-1] % head_dim != 0:
                    continue
                t = t.reshape(t.shape[0], t.shape[-1] // head_dim, head_dim)

            out[f"{prefix}_last"] = t[-1].tolist()
            out[f"{prefix}_mean"] = t.mean(0).tolist()

        out_rows.append(out)

    return out_rows


def run_local_census(cfg: AtlasRunConfig, hf_token: str | None = None) -> dict[int, int]:
    """Capture multi-layer activations into `l<N>_census_raw.json` files.

    GPU-optimized: keeps tensors on GPU during the forward pass, transfers the
    whole batch to CPU once, vectorizes per-layer slicing, and writes layer
    streams in parallel threads.
    """
    import torch
    from tqdm import tqdm

    corpus = list(iter_jsonl(cfg.corpus.path))
    if not corpus:
        raise ValueError(f"Corpus is empty: {cfg.corpus.path}")
    missing_prompt = [i for i, row in enumerate(corpus) if cfg.corpus.prompt_key not in row]
    if missing_prompt:
        raise ValueError(f"Corpus rows missing {cfg.corpus.prompt_key!r}: first bad row {missing_prompt[0]}")

    print(f"[extract] loading model: {cfg.model.model_id}")
    model, tokenizer = _load_model_and_tokenizer(cfg, hf_token)
    layers = resolve_layers(model)
    text_cfg = getattr(model.config, "text_config", model.config)

    per_layer_info: dict[int, dict] = {}
    for layer in cfg.layers:
        if layer >= len(layers):
            print(f"[extract] [warn] layer {layer} out of range; model has {len(layers)} layers")
            continue
        info = inspect_layer(layers[layer], text_cfg)
        if info["mlp"]["down_proj"] is None:
            print(f"[extract] [warn] layer {layer} has no MLP down projection; skipping")
            continue
        per_layer_info[layer] = info
        mlp, attn = info["mlp"], info["attn"]
        print(
            f"[extract] layer {layer:>2}: d_mlp={mlp['d_mlp']} "
            f"Q={attn['n_heads']}x{attn['head_dim']} KV={attn['n_kv_heads']}x{attn['head_dim']} "
            f"attn={attn['class_name']}"
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
            print(f"[extract] truncated forward graph to {deepest + 1} layers")

    cfg.outdir.mkdir(parents=True, exist_ok=True)
    streams = {
        layer: write_json_array_stream(cfg.outdir / f"l{layer}_census_raw.json")
        for layer in per_layer_info
    }
    counts = {layer: 0 for layer in per_layer_info}

    # Pre-bake per-row metadata so the hot loop is pure tensor work.
    base_rows = [
        {
            "_record_idx": i,
            "_bucket": _bucket_for(row, cfg.corpus.category_key, cfg.corpus.bucket_key),
            cfg.corpus.category_key: row.get(cfg.corpus.category_key, ""),
            "subcategory": row.get("subcategory", ""),
            cfg.corpus.prompt_key: row[cfg.corpus.prompt_key],
            "is_contrast": row.get("is_contrast", False),
            "contrast_pair_id": row.get("contrast_pair_id"),
            "id": row.get("id"),
        }
        for i, row in enumerate(corpus)
    ]

    # One executor for the whole run. We parallelize the CPU-heavy row slicing
    # across layers; JSON writes stay in the main thread to avoid stream corruption.
    max_workers = min(len(per_layer_info), 8)

    with streams_context(streams) as writers, concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for start in tqdm(range(0, len(corpus), cfg.batch_size), desc="batches", unit="batch"):
            batch_rows = base_rows[start:start + cfg.batch_size]
            prompts = [row[cfg.corpus.prompt_key] for row in batch_rows]
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

            # Build all layer rows in parallel (CPU-bound numpy/tensor slicing).
            def _build_for_layer(layer):
                return layer, _build_layer_rows(
                    layer=layer,
                    info=per_layer_info[layer],
                    captured=captured,
                    rows=batch_rows,
                    seq_lens=seq_lens,
                    cfg=cfg,
                )

            layer_rows = dict(pool.map(_build_for_layer, per_layer_info.keys()))

            # Write sequentially so the JSON streams stay consistent.
            for layer in per_layer_info:
                writer = writers[layer]
                for out in layer_rows[layer]:
                    writer.write(out)
                counts[layer] += len(layer_rows[layer])

            captured.clear()
            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()

    print(f"[extract] wrote: {counts}")
    return counts


class streams_context:
    def __init__(self, streams: dict[int, Any]):
        self.streams = streams
        self.opened: dict[int, Any] = {}

    def __enter__(self):
        self.opened = {layer: stream.__enter__() for layer, stream in self.streams.items()}
        return self.opened

    def __exit__(self, exc_type, exc, tb):
        for stream in self.streams.values():
            stream.__exit__(exc_type, exc, tb)
