from __future__ import annotations

import concurrent.futures
from typing import Any

from qwip_atlas.config import AtlasRunConfig
from qwip_atlas.io import iter_jsonl, write_npz_array_stream
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


def _register_hooks(per_layer_info: dict[int, dict], captured: dict[tuple[int, str], Any], components: set[str]):
    """Register hooks only for the components the user asked for.

    This avoids materializing large activation tensors that will be thrown away.
    mlp_hidden is always captured because it drives the base metadata field
    max_token_idx and the legacy 'mlp' component.
    """
    handles = []

    def make_hook(layer: int, key: str, take_input: bool = False):
        if take_input:
            def _hook(module, inputs):
                captured[(layer, key)] = inputs[0].detach()
            return _hook

        def _hook(module, inputs, output):
            x = output[0] if isinstance(output, tuple) else output
            captured[(layer, key)] = x.detach()
        return _hook

    for layer, info in per_layer_info.items():
        mlp, attn = info["mlp"], info["attn"]
        # Always need mlp_hidden for metadata max_token_idx.
        if mlp["down_proj"] is not None:
            handles.append(mlp["down_proj"].register_forward_pre_hook(
                make_hook(layer, "mlp_hidden", take_input=True)
            ))
        if "gate" in components and mlp["gate_proj"] is not None:
            handles.append(mlp["gate_proj"].register_forward_hook(make_hook(layer, "gate_pre")))
        if "up" in components and mlp["up_proj"] is not None:
            handles.append(mlp["up_proj"].register_forward_hook(make_hook(layer, "up")))
        if "q" in components and attn["q_proj"] is not None:
            handles.append(attn["q_proj"].register_forward_hook(make_hook(layer, "q")))
        if "k" in components and attn["k_proj"] is not None:
            handles.append(attn["k_proj"].register_forward_hook(make_hook(layer, "k")))
        if "v" in components and attn["v_proj"] is not None:
            handles.append(attn["v_proj"].register_forward_hook(make_hook(layer, "v")))
        if "heads" in components and attn["o_proj"] is not None:
            handles.append(attn["o_proj"].register_forward_pre_hook(
                make_hook(layer, "attn_pre", take_input=True)
            ))
        if "attn" in components and attn["module"] is not None:
            handles.append(attn["module"].register_forward_hook(make_hook(layer, "attn_out")))

    return handles


def _slice_and_mean(tensor: Any, seq_lens: list[int]) -> tuple[Any, Any]:
    """Vectorized last-token and variable-length mean over a batch.

    Input tensor has shape [B, max_seq_len, ...] after slicing to the longest
    real sequence. Padding is left-aligned, so real tokens are at the end.
    Returns (last_token, mean_tokens) both shaped [B, ...].
    """
    import numpy as np

    B = len(seq_lens)
    max_seq_len = tensor.shape[1]
    last = tensor[:, -1, ...]

    # Build a per-example length mask.
    mask = np.zeros((B, max_seq_len), dtype=bool)
    for i, length in enumerate(seq_lens):
        mask[i, -length:] = True

    # Expand mask to broadcast against arbitrary trailing dims.
    expand_axes = tuple(range(2, tensor.ndim))
    if expand_axes:
        mask = np.expand_dims(mask, axis=expand_axes)
    masked = tensor * mask
    summed = masked.sum(axis=1)
    mean = summed / np.array(seq_lens).reshape((B,) + (1,) * (tensor.ndim - 2))
    return last, mean


def _build_layer_arrays(
    layer: int,
    info: dict,
    captured: dict[tuple[int, str], Any],
    rows: list[dict[str, Any]],
    seq_lens: list[int],
    cfg: AtlasRunConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one batch of census arrays for a single layer.

    Returns (metadata_list, arrays_dict). Arrays are numpy arrays with no
    .tolist() conversion; the writer concatenates them directly into .npz.
    """
    import numpy as np
    import torch

    mlp_hidden = captured.get((layer, "mlp_hidden"))
    if mlp_hidden is None:
        return [], {}

    act_fn = info["mlp"].get("act_fn") or torch.nn.functional.silu
    head_dim = info["attn"]["head_dim"]
    components = cfg.components

    use_cuda = mlp_hidden.device.type == "cuda"
    max_seq_len = max(seq_lens)
    sl = slice(-max_seq_len, None)

    # Slice away padding on GPU before any transfer.
    mlp_hidden = mlp_hidden[:, sl]

    # Apply silu to gate on GPU before transfer, then collect all needed tensors.
    gpu_tensors: dict[str, Any] = {"mlp_hidden": mlp_hidden}
    if "gate" in components and (layer, "gate_pre") in captured:
        gpu_tensors["gate"] = act_fn(captured[(layer, "gate_pre")][:, sl])
    for key, comp_key in [
        ("up", "up"),
        ("attn", "attn_out"),
        ("attn_heads", "attn_pre"),
        ("q_heads", "q"),
        ("k_heads", "k"),
        ("v_heads", "v"),
    ]:
        if key_to_component_name(comp_key) in components and (layer, comp_key) in captured:
            gpu_tensors[key] = captured[(layer, comp_key)][:, sl]

    # Transfer everything to CPU in one non-blocking wave, then synchronize once.
    cpu_tensors = {
        k: v.float().to("cpu", non_blocking=use_cuda)
        for k, v in gpu_tensors.items()
    }
    if use_cuda:
        torch.cuda.current_stream().synchronize()

    np_tensors = {k: v.numpy() for k, v in cpu_tensors.items()}
    mlp_np = np_tensors.pop("mlp_hidden")

    arrays: dict[str, Any] = {}

    # MLP component.
    if "mlp" in components:
        arrays["last_token"], arrays["mean_tokens"] = _slice_and_mean(mlp_np, seq_lens)

    # Non-per-head components.
    for key, out_prefix in [("gate", "gate"), ("up", "up"), ("attn", "attn")]:
        if key in np_tensors:
            last, mean = _slice_and_mean(np_tensors[key], seq_lens)
            arrays[f"{out_prefix}_last"] = last
            arrays[f"{out_prefix}_mean"] = mean

    # Per-head components: reshape [B, seq, H*Dh] -> [B, seq, H, Dh].
    for key, out_prefix in [
        ("attn_heads", "attn_heads"),
        ("q_heads", "q_heads"),
        ("k_heads", "k_heads"),
        ("v_heads", "v_heads"),
    ]:
        if key not in np_tensors:
            continue
        t = np_tensors[key]
        if head_dim and t.shape[-1] % head_dim == 0:
            t = t.reshape(*t.shape[:-1], t.shape[-1] // head_dim, head_dim)
            # Move head axis to be adjacent to batch for easier downstream reading.
            # Current: [B, seq, H, Dh]. Downstream expects records as [H, Dh].
            last, mean = _slice_and_mean(t, seq_lens)
            arrays[f"{out_prefix}_last"] = last
            arrays[f"{out_prefix}_mean"] = mean

    # Metadata for the batch.
    metadata = []
    abs_sum = np.abs(mlp_np).sum(axis=-1)
    for i, row in enumerate(rows):
        sl_i = slice(-int(seq_lens[i]), None)
        metadata.append({
            "id": row.get("id", f"p{row.get('_record_idx', i):06d}"),
            "bucket": row.get("_bucket", "uncategorized"),
            "category": row.get(cfg.corpus.category_key, ""),
            "subcategory": row.get("subcategory", ""),
            "prompt": row[cfg.corpus.prompt_key],
            "is_contrast": row.get("is_contrast", False),
            "contrast_pair_id": row.get("contrast_pair_id"),
            "seq_len": int(seq_lens[i]),
            "max_token_idx": int(abs_sum[i, sl_i].argmax()),
        })

    return metadata, arrays


def key_to_component_name(key: str) -> str:
    return {
        "gate_pre": "gate",
        "up": "up",
        "attn_out": "attn",
        "attn_pre": "heads",
        "q": "q",
        "k": "k",
        "v": "v",
    }.get(key, key)


def run_local_census(cfg: AtlasRunConfig, hf_token: str | None = None) -> dict[int, int]:
    """Capture multi-layer activations into `l<N>_census_raw.npz` files.

    GPU-optimized:
      - Hooks only for components the user wants.
      - Tensors stay on GPU during forward, slice to max real seq len, then
        transfer to CPU in one non-blocking wave per batch.
      - Last-token and variable-length mean are computed vectorized in numpy.
      - No per-row .tolist(); arrays are concatenated directly into .npz.
    """
    import time

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
        layer: write_npz_array_stream(cfg.outdir / f"l{layer}_census_raw.npz")
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

    # One executor for the whole run. We parallelize the CPU-heavy array slicing
    # across layers; writes stay in the main thread to keep file streams safe.
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

            t0 = time.time()
            captured: dict[tuple[int, str], Any] = {}
            handles = _register_hooks(per_layer_info, captured, cfg.components)
            with torch.no_grad():
                model(**enc, use_cache=False)
            for handle in handles:
                handle.remove()
            t_forward = time.time() - t0

            t0 = time.time()

            def _build_for_layer(layer):
                return layer, _build_layer_arrays(
                    layer=layer,
                    info=per_layer_info[layer],
                    captured=captured,
                    rows=batch_rows,
                    seq_lens=seq_lens,
                    cfg=cfg,
                )

            layer_batches = dict(pool.map(_build_for_layer, per_layer_info.keys()))
            t_build = time.time() - t0

            t0 = time.time()
            for layer in per_layer_info:
                metadata, arrays = layer_batches[layer]
                writers[layer].write(metadata, arrays)
                counts[layer] += len(metadata)
            t_write = time.time() - t0

            captured.clear()
            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()

            if start == 0 or (start // cfg.batch_size) % 5 == 0:
                print(
                    f"[timings] batch {start//cfg.batch_size}: "
                    f"forward={t_forward:.3f}s build={t_build:.3f}s write={t_write:.3f}s"
                )

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
        # Finalize each layer's .npz in parallel. Each finalize loads temp files,
        # concatenates, compresses, and writes — this is CPU/disk heavy and layers
        # are independent, so parallelizing gives a large wall-clock win.
        import concurrent.futures
        import time

        from tqdm import tqdm

        def _close_one(stream):
            t0 = time.time()
            stream.__exit__(exc_type, exc, tb)
            return time.time() - t0

        print(f"[finalize] writing {len(self.streams)} compressed .npz files in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(self.streams), 8)) as pool:
            futures = {pool.submit(_close_one, stream): name for name, stream in self.streams.items()}
            with tqdm(total=len(self.streams), unit="layer", desc="finalize") as pbar:
                for fut in concurrent.futures.as_completed(futures):
                    name = futures[fut]
                    elapsed = fut.result()
                    pbar.set_postfix({f"l{name}": f"{elapsed:.1f}s"})
                    pbar.update(1)
        print("[finalize] done")
