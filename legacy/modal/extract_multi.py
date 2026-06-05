"""
extract_multi.py
----------------
Multi-layer sibling to extract_l11.py. Captures the FULL component set
(mlp, gate, up, attn, heads, q, k, v) for many layers in a SINGLE Modal job
with one forward pass per batch. Output is one `l<N>_census_raw.json` per
layer, byte-compatible with analyze_l11.py and build_atlas.py merge-layer.

Use this when you want to fill multiple gaps in one shot:

    modal run extract_multi.py --layers 16,18-22,25-34

Compared to looping `extract_l11.py` per layer this is ~10× faster because
we pay the model-load + image cold-start cost ONCE and hook every layer in
the same forward pass.

Quality guarantees vs extract_l11.py:
  * same corpus (juiceb0xc0de/mapping-prompts/prompts.jsonl, 825 prompts)
  * same hook points and same dynamic _inspect_layer geometry detection
  * same per-record field set per layer
  * same auto-cutoff (no truncation when targets go past CUTOFF_LAYER)

Downstream:
    for N in 16 18 19 20 21 22 25 26 27 28 29 30 31 32 33 34; do
      python analyze_l11.py --layer $N
      python build_atlas.py merge-layer --layer $N \\
          --census l${N}_census_raw.json --analysis-dir .
    done
    python build_atlas.py index
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config (mirrors extract_l11.py)
# ---------------------------------------------------------------------------
MODEL_ID     = "Qwen/Qwen3-8B-Base"
CUTOFF_LAYER = 36            # base; expands to max(target_layers)+1 automatically
CORPUS_REPO  = "juiceb0xc0de/mapping-prompts"
CORPUS_FILE  = "prompts.jsonl"
GPU_TYPE     = "T4"
BATCH_SIZE   = 64
OUTPUT_DIR   = "/data/qwen3-8b-base"
COMMIT_EVERY = 8             # streaming: flush+commit every N batches (durable progress + frees staged disk)

app = modal.App("l-multi-census")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "gcc", "g++")
    .run_commands("python -m pip install --upgrade pip setuptools wheel")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "transformers>=5.5.0",
        "accelerate>=0.34.0",
        "numpy",
        "huggingface_hub",
        "hf_transfer",
        "sentencepiece",
        "protobuf",
        "tqdm",
        "orjson",
    )
    .env({
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)

def _download_model():
    from huggingface_hub import snapshot_download
    import os
    snapshot_download(MODEL_ID, token=os.environ.get("HF_TOKEN"))

image = image.run_function(
    _download_model,
    secrets=[modal.Secret.from_name("huggingface")],
)

volume = modal.Volume.from_name("training_data", create_if_missing=True)


# ---------------------------------------------------------------------------
# Helpers (duplicated from extract_l11.py — small, kept local for portability)
# ---------------------------------------------------------------------------

def _layers_container(model):
    import torch.nn as nn
    from collections import deque
    queue = deque([model])
    while queue:
        m = queue.popleft()
        layers = getattr(m, "layers", None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return m
        for _, child in m.named_children():
            queue.append(child)
    raise RuntimeError(f"Cannot find layers ModuleList on {type(model).__name__}")


def _resolve_layers(model):
    return list(_layers_container(model).layers)


def _inspect_layer(layer_mod, text_cfg) -> dict:
    info = {
        "mlp":  {"module": None, "down_proj": None, "gate_proj": None, "up_proj": None,
                 "act_fn": None, "d_mlp": None},
        "attn": {"module": None, "q_proj": None, "k_proj": None, "v_proj": None,
                 "o_proj": None, "n_heads": None, "n_kv_heads": None, "head_dim": None,
                 "class_name": None},
    }
    mlp = getattr(layer_mod, "mlp", None)
    if mlp is not None:
        info["mlp"]["module"]    = mlp
        info["mlp"]["down_proj"] = getattr(mlp, "down_proj", None)
        info["mlp"]["gate_proj"] = getattr(mlp, "gate_proj", None)
        info["mlp"]["up_proj"]   = getattr(mlp, "up_proj",   None)
        info["mlp"]["act_fn"]    = getattr(mlp, "act_fn",    None)
        if info["mlp"]["down_proj"] is not None:
            info["mlp"]["d_mlp"] = info["mlp"]["down_proj"].in_features

    attn = getattr(layer_mod, "self_attn", None) or getattr(layer_mod, "attention", None)
    if attn is not None:
        info["attn"]["module"]     = attn
        info["attn"]["class_name"] = type(attn).__name__
        info["attn"]["q_proj"]     = getattr(attn, "q_proj", None)
        info["attn"]["k_proj"]     = getattr(attn, "k_proj", None)
        info["attn"]["v_proj"]     = getattr(attn, "v_proj", None)
        info["attn"]["o_proj"]     = getattr(attn, "o_proj", None)

        head_dim = getattr(attn, "head_dim", None) or getattr(text_cfg, "head_dim", None)

        n_heads = None
        if info["attn"]["o_proj"] is not None and head_dim:
            n_heads = info["attn"]["o_proj"].in_features // head_dim
        elif info["attn"]["q_proj"] is not None and head_dim:
            n_heads = info["attn"]["q_proj"].out_features // head_dim
        else:
            n_heads = getattr(attn, "num_heads", None) or getattr(text_cfg, "num_attention_heads", None)

        n_kv_heads = None
        if info["attn"]["k_proj"] is not None and head_dim:
            n_kv_heads = info["attn"]["k_proj"].out_features // head_dim
        else:
            n_kv_heads = getattr(attn, "num_key_value_heads", None) or \
                         getattr(text_cfg, "num_key_value_heads", n_heads)

        info["attn"]["n_heads"]    = n_heads
        info["attn"]["n_kv_heads"] = n_kv_heads
        info["attn"]["head_dim"]   = head_dim
    return info


def parse_layer_spec(spec: str) -> list[int]:
    """'16,18-22,25-34' → [16,18,19,20,21,22,25,26,27,28,29,30,31,32,33,34]"""
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return sorted(set(out))


# ---------------------------------------------------------------------------
# Multi-layer census
# ---------------------------------------------------------------------------

def run_multi_census(
    corpus: list[dict],
    model_id: str,
    target_layers: list[int],
    cutoff_layer: int = CUTOFF_LAYER,
    batch_size: int = BATCH_SIZE,
    hf_token: str | None = None,
    stream_dir: str | None = None,
) -> dict[int, list[dict]] | dict[int, int]:
    """Forward-pass each prompt once; capture all hooks across every target_layer.

    Default: returns dict[layer] -> list of per-prompt records (held in RAM).
    stream_dir set: writes each batch's records straight to
    `{stream_dir}/l{N}_census_raw.json` (valid JSON array, built incrementally) and
    returns dict[layer] -> record count. RAM stays flat (~one batch) instead of
    growing with the corpus — required for the deepened ~5k-prompt corpus, which
    OOMs the accumulate-everything path at ~18 GB/layer."""
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    print(f"[multi] loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, token=hf_token,
        dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    layers   = _resolve_layers(model)
    n_layers = len(layers)
    text_cfg = getattr(model.config, "text_config", model.config)

    # Probe every target up front. Skip out-of-range.
    per_layer_info: dict[int, dict] = {}
    for L in target_layers:
        if L >= n_layers:
            print(f"[multi] [warn] layer {L} out of range (model has {n_layers}); skipping")
            continue
        info = _inspect_layer(layers[L], text_cfg)
        if info["mlp"]["down_proj"] is None:
            print(f"[multi] [warn] layer {L} has no mlp.down_proj; skipping")
            continue
        per_layer_info[L] = info
        m, a = info["mlp"], info["attn"]
        print(f"[multi] layer {L:>2}: d_mlp={m['d_mlp']}  "
              f"Q={a['n_heads']}x{a['head_dim']}  KV={a['n_kv_heads']}x{a['head_dim']}  "
              f"attn={a['class_name']}")

    if not per_layer_info:
        raise RuntimeError("no valid target layers")

    # Auto-expand cutoff so every target is reachable in the forward pass.
    deepest = max(per_layer_info.keys())
    effective_cutoff = max(cutoff_layer, deepest + 1)
    if effective_cutoff < n_layers:
        container = _layers_container(model)
        container.layers = container.layers[:effective_cutoff]
        torch.cuda.empty_cache()
        print(f"[multi] truncated to {effective_cutoff} layers (dropped {n_layers - effective_cutoff})")
    else:
        print(f"[multi] keeping all {n_layers} layers (deepest target = {deepest})")

    # Accumulator: stream to disk (flat RAM) or hold in memory (original behavior)
    streaming = stream_dir is not None
    if streaming:
        import orjson
        os.makedirs(stream_dir, exist_ok=True)
        _handles = {L: open(f"{stream_dir}/l{L}_census_raw.json", "wb") for L in per_layer_info}
        for _h in _handles.values():
            _h.write(b"[")
        _first  = {L: True for L in per_layer_info}
        _counts = {L: 0    for L in per_layer_info}
        per_layer_records = None
        print(f"[multi] STREAM mode — flushing per batch to {stream_dir} (flat RAM)")
    else:
        per_layer_records = {L: [] for L in per_layer_info}

    n_batches = (len(corpus) + batch_size - 1) // batch_size
    print(f"[multi] {len(corpus)} prompts → {n_batches} batches of {batch_size}  layers={len(per_layer_info)}")

    for batch_idx, batch_start in enumerate(tqdm(range(0, len(corpus), batch_size), desc="batches", unit="batch")):
        batch   = corpus[batch_start: batch_start + batch_size]
        prompts = [r["prompt"] for r in batch]
        print(f"[multi] batch {batch_idx+1}/{n_batches}  prompts {batch_start}–{batch_start+len(batch)-1}  encoding ...", flush=True)

        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=512)
        seq_lens = enc["attention_mask"].sum(dim=1).tolist()
        max_seq = max(seq_lens)
        print(f"[multi] batch {batch_idx+1}/{n_batches}  seq_lens min={min(seq_lens)} max={max_seq}  running forward pass ...", flush=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}

        captured: dict[tuple[int, str], "torch.Tensor"] = {}

        def make_hook(L, key, take_input=False):
            if take_input:
                def _h(module, inp):
                    captured[(L, key)] = inp[0].detach().float().cpu()
                return _h
            def _h(module, inp, out):
                x = out[0] if isinstance(out, tuple) else out
                captured[(L, key)] = x.detach().float().cpu()
            return _h

        handles = []
        for L, info in per_layer_info.items():
            m, a = info["mlp"], info["attn"]
            handles.append(m["down_proj"].register_forward_pre_hook(
                make_hook(L, "mlp_hidden", take_input=True)))
            if m["gate_proj"] is not None:
                handles.append(m["gate_proj"].register_forward_hook(make_hook(L, "gate_pre")))
            if m["up_proj"] is not None:
                handles.append(m["up_proj"].register_forward_hook(make_hook(L, "up")))
            if a["q_proj"] is not None:
                handles.append(a["q_proj"].register_forward_hook(make_hook(L, "q")))
            if a["k_proj"] is not None:
                handles.append(a["k_proj"].register_forward_hook(make_hook(L, "k")))
            if a["v_proj"] is not None:
                handles.append(a["v_proj"].register_forward_hook(make_hook(L, "v")))
            if a["o_proj"] is not None:
                handles.append(a["o_proj"].register_forward_pre_hook(
                    make_hook(L, "attn_pre", take_input=True)))
            if a["module"] is not None:
                handles.append(a["module"].register_forward_hook(make_hook(L, "attn_out")))

        with torch.no_grad():
            model(**enc, use_cache=False)

        for h in handles:
            h.remove()

        # Per-layer record assembly (same shape as extract_l11.py)
        for L, info in per_layer_info.items():
            act_fn   = info["mlp"]["act_fn"] or torch.nn.functional.silu
            head_dim = info["attn"]["head_dim"]
            mlp_hidden = captured.get((L, "mlp_hidden"))
            if mlp_hidden is None:
                continue
            B, T = mlp_hidden.shape[:2]

            gate_pre = captured.get((L, "gate_pre"))
            gate_post = act_fn(gate_pre) if gate_pre is not None else None
            up_out   = captured.get((L, "up"))
            attn_out = captured.get((L, "attn_out"))

            def _safe_per_head(t):
                if t is None or not head_dim or t.shape[-1] % head_dim != 0:
                    return None
                return t.reshape(B, T, t.shape[-1] // head_dim, head_dim)

            attn_heads = _safe_per_head(captured.get((L, "attn_pre")))
            q_heads    = _safe_per_head(captured.get((L, "q")))
            k_heads    = _safe_per_head(captured.get((L, "k")))
            v_heads    = _safe_per_head(captured.get((L, "v")))

            for i, (rec, seq_len) in enumerate(zip(batch, seq_lens)):
                seq_len = int(seq_len)
                sl = slice(-seq_len, None)
                mh = mlp_hidden[i, sl]

                row = {
                    "id":               rec.get("id", f"p{batch_start + i:04d}"),
                    "bucket":           rec.get("bucket", rec["category"].lower().replace(" & ", "_").replace(" ", "_")),
                    "category":         rec["category"],
                    "subcategory":      rec.get("subcategory", ""),
                    "prompt":           rec["prompt"],
                    "is_contrast":      rec.get("is_contrast", False),
                    "contrast_pair_id": rec.get("contrast_pair_id"),
                    "seq_len":          seq_len,
                    "max_token_idx":    int(mh.abs().sum(dim=-1).argmax().item()),
                    "last_token":       mh[-1].numpy().tolist(),
                    "mean_tokens":      mh.mean(0).numpy().tolist(),
                }

                def _add(prefix, tensor_per_seq):
                    if tensor_per_seq is None:
                        return
                    t = tensor_per_seq[i, sl]
                    row[f"{prefix}_last"] = t[-1].numpy().tolist()
                    row[f"{prefix}_mean"] = t.mean(0).numpy().tolist()

                _add("gate",        gate_post)
                _add("up",          up_out)
                _add("attn",        attn_out)
                _add("attn_heads",  attn_heads)
                _add("q_heads",     q_heads)
                _add("k_heads",     k_heads)
                _add("v_heads",     v_heads)

                if streaming:
                    h = _handles[L]
                    if not _first[L]:
                        h.write(b",")
                    h.write(orjson.dumps(row))
                    _first[L]  = False
                    _counts[L] += 1
                else:
                    per_layer_records[L].append(row)

        # Free the captured dict to keep CPU RAM steady across batches.
        captured.clear()

        # Checkpoint: flush + commit every COMMIT_EVERY batches so progress is durable
        # mid-run and staged writes don't pile up on container disk (the likely cause of
        # "Runner disappeared" near the end of a long, uncommitted ~300GB job).
        if streaming and (batch_idx + 1) % COMMIT_EVERY == 0:
            try:
                for _h in _handles.values():
                    _h.flush()
                volume.commit()
                print(f"[multi] checkpoint commit @ batch {batch_idx+1}/{n_batches}", flush=True)
            except Exception as e:
                print(f"[multi] [warn] checkpoint commit failed: {e}", flush=True)

    if streaming:
        for L, _h in _handles.items():
            _h.write(b"]")
            _h.close()
        print(f"[multi] streamed layers written: {sorted(_counts)}", flush=True)
        return _counts

    return per_layer_records


# ---------------------------------------------------------------------------
# Modal remote
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=86400,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def extract_remote(layers: list[int], batch_size: int = BATCH_SIZE,
                   stream: bool = False) -> dict[int, int]:
    import os
    from huggingface_hub import hf_hub_download

    hf_token = os.environ.get("HF_TOKEN")

    print(f"[multi] downloading corpus from {CORPUS_REPO}/{CORPUS_FILE} ...")
    local_path = hf_hub_download(
        repo_id=CORPUS_REPO, filename=CORPUS_FILE, repo_type="dataset", token=hf_token,
    )
    with open(local_path) as f:
        corpus = [json.loads(line) for line in f]
    print(f"[multi] loaded {len(corpus)} prompts; target layers={layers}")

    # STREAM path: records flushed per batch (flat RAM) — required for the ~5k corpus.
    if stream:
        counts = run_multi_census(corpus, MODEL_ID, layers, batch_size=batch_size,
                                  hf_token=None, stream_dir=OUTPUT_DIR)
        volume.commit()
        print(f"[multi] volume committed (streamed). layers: {sorted(counts.keys())}", flush=True)
        return counts

    results = run_multi_census(corpus, MODEL_ID, layers,
                               batch_size=batch_size, hf_token=None)

    # Write all layer files in parallel then commit once
    import concurrent.futures
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def _write_layer(item):
        import orjson
        L, records = item
        out = f"{OUTPUT_DIR}/l{L}_census_raw.json"
        with open(out, "wb") as f:
            f.write(orjson.dumps(records))
        print(f"[multi] wrote {len(records)} records → {out}", flush=True)
        return L, len(records)

    print(f"[multi] writing {len(results)} layer files in parallel ...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(results)) as ex:
        counts = dict(ex.map(_write_layer, results.items()))

    volume.commit()
    print(f"[multi] volume committed. layers written: {sorted(counts.keys())}", flush=True)
    return counts


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(layers: str, batch_size: int = BATCH_SIZE, stream: bool = False):
    layer_list = parse_layer_spec(layers)
    mode = "STREAM (flat RAM)" if stream else "in-RAM (original)"
    print(f"[local] requesting layers: {layer_list}  batch_size={batch_size}  mode={mode}")
    summary = extract_remote.remote(layer_list, batch_size, stream)

    # Hint for the next step
    layers_str = " ".join(str(L) for L in sorted(summary.keys()))
    print()
    print("Next steps:")
    print(f"  for N in {layers_str}; do")
    print(f"    python analyze_l11.py --layer $N --no-auto-extract")
    print(f"    python build_atlas.py merge-layer --layer $N \\")
    print(f"        --census l${{N}}_census_raw.json --analysis-dir .")
    print(f"  done")
    print(f"  python build_atlas.py index")


if __name__ == "__main__":
    print("Run with:  modal run extract_multi.py --layers 16,18-22,25-34")
    raise SystemExit(0)