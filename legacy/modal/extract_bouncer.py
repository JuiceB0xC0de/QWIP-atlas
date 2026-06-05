"""
extract_bouncer.py
------------------
Captures corporate-vs-authentic axis activations across a list of layers in
ONE forward pass per batch. Computes per-(layer, component, feature) F-stat
and corporate-minus-authentic delta for the bouncer-axis bouncer map.

This is the complement to extract_l11.py:
  extract_l11.py     — single layer, 16-bucket topic structure
  extract_bouncer.py — many layers, binary corporate/authentic axis

The output is a single JSON keyed by layer -> component -> stat-arrays, ready
to be folded into the master atlas with:

  python build_atlas.py merge-bouncer --report bouncer_scores.json

Usage:
  modal run extract_bouncer.py --layers 0-15
  modal run extract_bouncer.py --layers 0,5,11,15,23
  python extract_bouncer.py --local --layers 0,11 --n 32
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID         = "Qwen/Qwen3-8B-Base"
CORPORA_DIR      = "/Users/chiggy/sub-zero/corpora"   # local source
CORPORATE_FILE   = "corporate_stems.jsonl"
AUTHENTIC_FILE   = "authentic_bella_samples.jsonl"
GPU_TYPE         = "H200"
BATCH_SIZE       = 100
OUTPUT_FILE      = "qwen3-8b-base/bouncer_scores.json"

app = modal.App("bouncer-extract")

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
    )
    .env({
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)

volume = modal.Volume.from_name("training_data", create_if_missing=True)


# ---------------------------------------------------------------------------
# Helpers (duplicated from extract_l11.py — small enough to keep local)
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


# ---------------------------------------------------------------------------
# Layer-spec parser:  "0-15" or "0,5,11" or "11"
# ---------------------------------------------------------------------------

def parse_layer_spec(spec: str) -> list[int]:
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif chunk:
            out.append(int(chunk))
    # Dedup + sort
    return sorted(set(out))


# ---------------------------------------------------------------------------
# Core: capture activations + compute bouncer scores
# ---------------------------------------------------------------------------

def run_bouncer_census(
    corp_prompts: list[str],
    auth_prompts: list[str],
    model_id: str,
    target_layers: list[int],
    batch_size: int = BATCH_SIZE,
    hf_token: str | None = None,
) -> dict:
    """Run one forward pass per batch with hooks on every target layer.
    Accumulate last-token activations per (layer, component) across all prompts.
    Compute binary F-statistic + corporate-minus-authentic delta per feature.
    Returns: dict[layer_idx] -> dict[component_name] -> stat dict."""
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    all_prompts = corp_prompts + auth_prompts
    labels = np.array(["corporate"] * len(corp_prompts) + ["authentic"] * len(auth_prompts))
    n_total = len(all_prompts)
    print(f"[bouncer] {len(corp_prompts)} corporate + {len(auth_prompts)} authentic = {n_total} prompts")
    print(f"[bouncer] target layers: {target_layers}")

    print(f"[bouncer] loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, token=hf_token,
        dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    layers = _resolve_layers(model)
    text_cfg = getattr(model.config, "text_config", model.config)

    # Inspect every target layer up front and store per-layer geometry.
    per_layer_info: dict[int, dict] = {}
    for L in target_layers:
        if L >= len(layers):
            print(f"[bouncer] [warn] layer {L} out of range (model has {len(layers)}); skipping")
            continue
        per_layer_info[L] = _inspect_layer(layers[L], text_cfg)
        m, a = per_layer_info[L]["mlp"], per_layer_info[L]["attn"]
        print(f"[bouncer] layer {L:>2}: d_mlp={m['d_mlp']}  "
              f"Q={a['n_heads']}x{a['head_dim']}  KV={a['n_kv_heads']}x{a['head_dim']}  "
              f"attn={a['class_name']}")

    # Storage: per (layer, component) -> running list of [n_features] arrays per prompt.
    # We accumulate last-token activations across batches and stack at the end.
    # Mem footprint: ~285 * sum_layer (d_mlp*3 + d_model + n_heads*head_dim*3) ≈ 100-300 MB.
    capture_store: dict[tuple[int, str], list] = {}

    def _ensure(L, comp):
        capture_store.setdefault((L, comp), [])

    for batch_start in tqdm(range(0, n_total, batch_size), desc="batches", unit="batch"):
        prompts = all_prompts[batch_start: batch_start + batch_size]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=512)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        seq_lens = enc["attention_mask"].sum(dim=1).tolist()

        # Per-layer captured tensors for this batch
        captured: dict[tuple[int, str], "torch.Tensor"] = {}

        def make_hook(L, comp, take_input=False):
            if take_input:
                def _h(module, inp):
                    captured[(L, comp)] = inp[0].detach().float().cpu()
                return _h
            def _h(module, inp, out):
                x = out[0] if isinstance(out, tuple) else out
                captured[(L, comp)] = x.detach().float().cpu()
            return _h

        handles = []
        for L, info in per_layer_info.items():
            m, a = info["mlp"], info["attn"]
            if m["down_proj"] is not None:
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

        # Pull last-token activations per prompt.
        # Left-padded input means real tokens occupy the tail, so index -1 is always real.
        for L in per_layer_info:
            for raw_key, store_key in [("mlp_hidden", "mlp"),
                                       ("gate_pre",   "gate_pre"),
                                       ("up",         "up"),
                                       ("attn_out",   "attn"),
                                       ("attn_pre",   "attn_heads"),
                                       ("q",          "q_heads"),
                                       ("k",          "k_heads"),
                                       ("v",          "v_heads")]:
                t = captured.get((L, raw_key))
                if t is None:
                    continue
                vec = t[:, -1, :].numpy()  # [batch, features_flat]
                _ensure(L, store_key)
                capture_store[(L, store_key)].append(vec)

    # Now stack and compute stats per (layer, component).
    print("[bouncer] computing F-stat + corp-auth delta ...")
    result: dict = {}
    for (L, comp), chunks in capture_store.items():
        A = np.concatenate(chunks, axis=0)  # [n_total, features]
        assert A.shape[0] == n_total, f"len mismatch for L{L}/{comp}: {A.shape[0]} != {n_total}"

        # gate_pre -> apply SiLU on the stacked array (cheaper than during capture)
        if comp == "gate_pre":
            comp = "gate"
            # silu(x) = x * sigmoid(x)
            x = A
            A = x * (1.0 / (1.0 + np.exp(-x)))

        n_c = len(corp_prompts)
        n_a = len(auth_prompts)
        x_c = A[:n_c]
        x_a = A[n_c:n_c + n_a]
        mean_c = x_c.mean(axis=0)
        mean_a = x_a.mean(axis=0)
        std_c  = x_c.std(axis=0)
        std_a  = x_a.std(axis=0)
        delta  = mean_c - mean_a

        # Binary one-way ANOVA F-stat:
        # between_var = (n_c*(mean_c - grand)**2 + n_a*(mean_a - grand)**2) / df_b
        # within_var  = (sum((x_c - mean_c)**2) + sum((x_a - mean_a)**2)) / df_w
        # F = between / within
        grand   = (n_c * mean_c + n_a * mean_a) / (n_c + n_a)
        between = (n_c * (mean_c - grand) ** 2 + n_a * (mean_a - grand) ** 2)
        within  = ((x_c - mean_c) ** 2).sum(axis=0) + ((x_a - mean_a) ** 2).sum(axis=0)
        df_b    = 1
        df_w    = n_c + n_a - 2
        fstat   = (between / df_b) / (within / df_w + 1e-10)

        layer_key = str(L)
        result.setdefault(layer_key, {})
        result[layer_key][comp] = {
            "n_corporate":   int(n_c),
            "n_authentic":   int(n_a),
            "n_features":    int(A.shape[1]),
            "mean_corp":     mean_c.astype("float32").tolist(),
            "mean_auth":     mean_a.astype("float32").tolist(),
            "std_corp":      std_c.astype("float32").tolist(),
            "std_auth":      std_a.astype("float32").tolist(),
            "delta":         delta.astype("float32").tolist(),
            "fstat":         fstat.astype("float32").tolist(),
            # geometry hints for downstream per-head reshape:
            "is_per_head":   comp in ("attn_heads", "q_heads", "k_heads", "v_heads"),
            "head_dim":      per_layer_info[L]["attn"]["head_dim"],
        }

    # Top-line summary printed inline
    print()
    print("[bouncer] top 3 features per (layer, component) by F-stat:")
    print("  layer | component   | top f-stat | mean delta_top3")
    print("  ------|-------------|------------|----------------")
    for L in sorted(result.keys(), key=int):
        for comp, data in result[L].items():
            fstats = np.asarray(data["fstat"])
            deltas = np.asarray(data["delta"])
            top3 = np.argsort(fstats)[-3:][::-1]
            top_score = float(fstats[top3[0]])
            mean_delta = float(deltas[top3].mean())
            print(f"  {L:>5} | {comp:<11} | {top_score:>10.3f} | {mean_delta:>+15.4f}")

    return result


# ---------------------------------------------------------------------------
# Modal remote
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=3600,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def extract_remote(corp_prompts: list[str], auth_prompts: list[str], layer_list: list[int]):
    hf_token = os.environ.get("HF_TOKEN")
    result = run_bouncer_census(corp_prompts, auth_prompts, MODEL_ID, layer_list, hf_token=hf_token)

    out = f"/data/{OUTPUT_FILE}"
    with open(out, "w") as f:
        json.dump(result, f)
    volume.commit()
    print(f"[bouncer] wrote scores for {len(result)} layers to {out}")
    return result


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

def _load_corpora(corpora_dir: str) -> tuple[list[str], list[str]]:
    """Read jsonl files and return (corporate_prompts, authentic_prompts)."""
    def _read(p: Path) -> list[str]:
        with open(p) as f:
            return [json.loads(line)["text"] for line in f if line.strip()]
    cd = Path(corpora_dir)
    corp = _read(cd / CORPORATE_FILE)
    auth = _read(cd / AUTHENTIC_FILE)
    return corp, auth


@app.local_entrypoint()
def main(layers: str = "0-15", corpora_dir: str = CORPORA_DIR):
    layer_list = parse_layer_spec(layers)
    corp, auth = _load_corpora(corpora_dir)
    print(f"[local] layers={layer_list}  corporate={len(corp)}  authentic={len(auth)}")
    result = extract_remote.remote(corp, auth, layer_list)
    print(f"Saved bouncer scores for {len(result)} layers to volume:/data/{OUTPUT_FILE}")


# ---------------------------------------------------------------------------
# Quick local test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--layers", default="0,11")
    parser.add_argument("--corpora-dir", default=CORPORA_DIR)
    parser.add_argument("--n", type=int, default=16,
                        help="Local test: cap prompts per bucket to this many")
    args = parser.parse_args()

    if not args.local:
        print("Run with:   modal run extract_bouncer.py --layers 0-15")
        print("Local test: python extract_bouncer.py --local --layers 0,11 --n 16")
        raise SystemExit(0)

    layer_list = parse_layer_spec(args.layers)
    corp, auth = _load_corpora(args.corpora_dir)
    corp = corp[: args.n]
    auth = auth[: args.n]
    result = run_bouncer_census(corp, auth, MODEL_ID, layer_list)
    out_path = Path("bouncer_scores_test.json")
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"Local test: wrote scores for {len(result)} layers to {out_path}")
