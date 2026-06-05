from __future__ import annotations

from pathlib import Path
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
        "dtype": _torch_dtype(model_spec.dtype),
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
                captured[(layer, key)] = inputs[0].detach().float().cpu()
            return _hook

        def _hook(module, inputs, output):
            x = output[0] if isinstance(output, tuple) else output
            captured[(layer, key)] = x.detach().float().cpu()
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


def _row_from_capture(
    *,
    row: dict[str, Any],
    record_idx: int,
    seq_len: int,
    layer: int,
    info: dict,
    captured: dict[tuple[int, str], Any],
    batch_index: int,
    cfg: AtlasRunConfig,
) -> dict[str, Any] | None:
    import torch

    mlp_hidden = captured.get((layer, "mlp_hidden"))
    if mlp_hidden is None:
        return None

    act_fn = info["mlp"]["act_fn"] or torch.nn.functional.silu
    head_dim = info["attn"]["head_dim"]
    batch_size, _, _ = mlp_hidden.shape
    sl = slice(-seq_len, None)
    mh = mlp_hidden[batch_index, sl]

    out = {
        "id": row.get("id", f"p{record_idx:06d}"),
        "bucket": _bucket_for(row, cfg.corpus.category_key, cfg.corpus.bucket_key),
        "category": row.get(cfg.corpus.category_key, ""),
        "subcategory": row.get("subcategory", ""),
        "prompt": row[cfg.corpus.prompt_key],
        "is_contrast": row.get("is_contrast", False),
        "contrast_pair_id": row.get("contrast_pair_id"),
        "seq_len": int(seq_len),
        "max_token_idx": int(mh.abs().sum(dim=-1).argmax().item()),
        "last_token": mh[-1].numpy().tolist(),
        "mean_tokens": mh.mean(0).numpy().tolist(),
    }

    gate_pre = captured.get((layer, "gate_pre"))
    gate_post = act_fn(gate_pre) if gate_pre is not None else None

    def safe_per_head(tensor):
        if tensor is None or not head_dim or tensor.shape[-1] % head_dim != 0:
            return None
        return tensor.reshape(batch_size, tensor.shape[1], tensor.shape[-1] // head_dim, head_dim)

    components = cfg.components
    tensors = {
        "gate": gate_post,
        "up": captured.get((layer, "up")),
        "attn": captured.get((layer, "attn_out")),
        "attn_heads": safe_per_head(captured.get((layer, "attn_pre"))),
        "q_heads": safe_per_head(captured.get((layer, "q"))),
        "k_heads": safe_per_head(captured.get((layer, "k"))),
        "v_heads": safe_per_head(captured.get((layer, "v"))),
    }
    key_to_component = {
        "gate": "gate",
        "up": "up",
        "attn": "attn",
        "attn_heads": "heads",
        "q_heads": "q",
        "k_heads": "k",
        "v_heads": "v",
    }

    for prefix, tensor in tensors.items():
        if tensor is None or key_to_component[prefix] not in components:
            continue
        t = tensor[batch_index, sl]
        out[f"{prefix}_last"] = t[-1].numpy().tolist()
        out[f"{prefix}_mean"] = t.mean(0).numpy().tolist()

    return out


def run_local_census(cfg: AtlasRunConfig, hf_token: str | None = None) -> dict[int, int]:
    """Capture multi-layer activations into `l<N>_census_raw.json` files.

    This is the model-agnostic, non-Modal replacement for the old `extract_multi.py`.
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

    with streams_context(streams) as writers:
        for start in tqdm(range(0, len(corpus), cfg.batch_size), desc="batches", unit="batch"):
            batch = corpus[start:start + cfg.batch_size]
            prompts = [row[cfg.corpus.prompt_key] for row in batch]
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
                for batch_index, (row, seq_len) in enumerate(zip(batch, seq_lens)):
                    out = _row_from_capture(
                        row=row,
                        record_idx=start + batch_index,
                        seq_len=int(seq_len),
                        layer=layer,
                        info=info,
                        captured=captured,
                        batch_index=batch_index,
                        cfg=cfg,
                    )
                    if out is not None:
                        writers[layer].write(out)
                        counts[layer] += 1

            captured.clear()

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
