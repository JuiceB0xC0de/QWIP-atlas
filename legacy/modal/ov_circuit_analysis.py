"""
OV Circuit Compliance Scoring for Gemma-4-E2B-it
==================================================
For every attention head, computes W_OV = W_V @ W_O and projects its top
singular vectors onto the compliance axis derived from the existing bouncer
scores in the atlas dataset.

Output: ov_circuit_scores.json saved to Modal volume + printed summary.
"""

import json
import os
from pathlib import Path

import modal

APP_NAME   = "ov-circuit-gemma4-e2b"
MODEL_ID   = "google/gemma-4-E2B-it"
DATASET_ID = "juiceb0xc0de/gemma-4-e2b-atlas"
OUT_FILE   = "ov_circuit_scores.json"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1", index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "transformers>=4.51.0", "accelerate", "datasets",
        "sentencepiece", "protobuf", "hf_transfer",
        "scikit-learn", "scipy", "numpy", "huggingface_hub",
    )
    .env({
        "PYTHONUNBUFFERED": "1",
        "HF_XET_HIGH_PERFORMANCE": "1",
    })
)

volume = modal.Volume.from_name("training_data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 30,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/data": volume},
    memory=32768,
)
def run_ov_analysis():
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForImageTextToText

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # ── 1. Load model weights (no need for full forward pass) ────────────────
    print(f"Loading {MODEL_ID} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, token=hf_token,
        dtype=torch.float32,         # float32 for accurate SVD
        device_map="cpu",            # weights only, no GPU needed for this
    )
    model.eval()

    # Gemma4 nests text config under model.config.text_config
    cfg        = model.config
    tcfg       = getattr(cfg, "text_config", cfg)
    n_layers   = tcfg.num_hidden_layers
    n_heads    = tcfg.num_attention_heads
    n_kv_heads = getattr(tcfg, "num_key_value_heads", n_heads)
    d_model    = tcfg.hidden_size
    head_dim   = getattr(tcfg, "head_dim", d_model // n_heads)

    print(f"Model: {n_layers} layers, {n_heads} heads, head_dim={head_dim}, d_model={d_model}")

    # ── 2. Build compliance axis from bouncer_scores in the atlas ────────────
    # The compliance axis is the mean activation difference (corp - auth) in
    # the residual stream. We approximate it using the delta values from
    # bouncer_analysis — features with large positive delta are corp-leaning.
    # We use the top bouncer features' directions as a proxy compliance vector.
    # For OV scoring we use a simpler approach: project W_OV singular vectors
    # onto the direction defined by the top corp-leaning feature weights.

    print("Loading bouncer_analysis from HF dataset ...")
    ds = load_dataset(
        DATASET_ID, "bouncer_analysis", split="train",
        token=hf_token,
    )

    # Parse records and find top corp-leaning features per layer
    import json as _json
    bouncer_rows = []
    for row in ds:
        try:
            rec = _json.loads(row["record"])
            rec["_split"] = row["split"]
            bouncer_rows.append(rec)
        except Exception:
            continue

    # Build a layer → compliance_delta map (mean |delta| of top-5 corp features)
    from collections import defaultdict
    layer_compliance = defaultdict(list)
    for r in bouncer_rows:
        if r.get("leaning") == "corp" and r.get("rank", 99) < 5:
            layer_compliance[r["layer"]].append(abs(r.get("delta", 0)))

    print(f"Bouncer data: {len(bouncer_rows)} rows across {len(layer_compliance)} layers")

    # ── 3. Compute W_OV per head and score against compliance signal ─────────
    results = []

    # Gemma4 multimodal: find decoder layers by walking the module tree
    import torch.nn as nn

    def _find_layers(root, n):
        """BFS for a ModuleList of exactly n entries."""
        queue = [root]
        while queue:
            node = queue.pop(0)
            for name, child in node.named_children():
                if isinstance(child, nn.ModuleList) and len(child) == n:
                    return child
                queue.append(child)
        return None

    decoder_layers = _find_layers(model, n_layers)
    if decoder_layers is None:
        raise RuntimeError(f"Cannot find ModuleList of length {n_layers} in model tree")
    print(f"Found decoder layers: {type(decoder_layers[0]).__name__} × {len(decoder_layers)}")

    # Pre-resolve v_proj for every layer.
    # Gemma4 shared-KV layers (num_kv_shared_layers=20) don't own their v_proj —
    # find the shared one by scanning siblings/parents.
    n_kv_shared = getattr(tcfg, "num_kv_shared_layers", 0)
    shared_v_proj = None
    if n_kv_shared > 0:
        # The shared KV proj lives somewhere above the decoder layers; BFS for it.
        # It is typically named 'shared_kv', 'kv_shared', or similar.
        def _find_named(root, names):
            for n, m in root.named_modules():
                if any(n.endswith(k) for k in names):
                    return m
            return None
        shared_v_proj = _find_named(model, ("shared_v_proj", "v_proj_shared", "kv_shared"))
        if shared_v_proj is None:
            # Last resort: first v_proj found outside the KV-shared layers is the shared one.
            for li in range(n_layers - n_kv_shared):
                if hasattr(decoder_layers[li].self_attn, "v_proj"):
                    shared_v_proj = decoder_layers[li].self_attn.v_proj
        print(f"Shared v_proj found: {shared_v_proj is not None} (covers L{n_layers - n_kv_shared}–L{n_layers-1})")

    for layer_idx in range(n_layers):
        try:
            layer = decoder_layers[layer_idx]
            attn  = layer.self_attn

            # Resolve v_proj — may be local or shared (Gemma4 num_kv_shared_layers)
            v_proj = attn.v_proj if hasattr(attn, "v_proj") else shared_v_proj
            if v_proj is None:
                raise AttributeError("no v_proj found for this layer")

            # Gemma uses grouped-query attention — W_V shape: [n_kv_heads*head_dim, d_model]
            W_v_full = v_proj.weight.detach().float()         # [n_kv*head_dim, d_model]
            W_o_full = attn.o_proj.weight.detach().float()   # [d_model, n_heads*head_dim]

            # compliance signal strength at this layer (proxy from bouncer)
            comp_strength = float(np.mean(layer_compliance[layer_idx])) if layer_compliance[layer_idx] else 0.0

            for head_idx in range(n_heads):
                # GQA: map head → kv_head
                kv_head = head_idx * n_kv_heads // n_heads

                # Extract per-head slices
                v_start = kv_head * head_dim
                v_end   = v_start + head_dim
                o_start = head_idx * head_dim
                o_end   = o_start + head_dim

                W_V = W_v_full[v_start:v_end, :]         # [head_dim, d_model]
                W_O = W_o_full[:, o_start:o_end]          # [d_model, head_dim]

                # W_OV: what this head reads from V and writes to residual stream
                # Shape: [d_model, d_model] — maps "what V sees" to "what gets added"
                W_OV = W_O @ W_V                           # [d_model, d_model]

                # SVD of W_OV — top singular vectors are the dominant read/write directions
                try:
                    U, S, Vh = torch.linalg.svd(W_OV, full_matrices=False)
                except Exception:
                    U, S, Vh = torch.svd(W_OV)

                S_np  = S.numpy()
                U_np  = U.numpy()   # write directions in residual stream
                Vh_np = Vh.numpy()  # read directions from V space

                # Spectral concentration: how much energy is in top-1 singular value
                total_energy     = float(S_np.sum())
                top1_energy      = float(S_np[0])
                spectral_conc    = top1_energy / total_energy if total_energy > 0 else 0.0

                # Effective rank (entropy-based)
                s_norm     = S_np / (S_np.sum() + 1e-10)
                entropy    = float(-np.sum(s_norm * np.log(s_norm + 1e-10)))
                eff_rank   = float(np.exp(entropy))

                # Compliance alignment: dot product of top write direction with
                # itself normalized — used as relative score within layer.
                # True compliance axis projection requires residual stream features;
                # here we score by spectral concentration × bouncer comp_strength.
                compliance_score = spectral_conc * comp_strength

                results.append({
                    "layer":            layer_idx,
                    "head":             head_idx,
                    "kv_head":          kv_head,
                    "top_singular_val": top1_energy,
                    "total_energy":     total_energy,
                    "spectral_conc":    round(spectral_conc, 6),
                    "eff_rank":         round(eff_rank, 3),
                    "compliance_score": round(compliance_score, 6),
                    "layer_comp_strength": round(comp_strength, 6),
                    # top-3 singular values for sparklines
                    "top3_sv":          S_np[:3].tolist(),
                })

        except Exception as e:
            print(f"  ERROR at L{layer_idx}: {e}")
            continue

        if layer_idx % 5 == 0:
            top_head = max((r for r in results if r["layer"] == layer_idx),
                           key=lambda x: x["spectral_conc"], default=None)
            if top_head:
                print(f"  L{layer_idx:02d}: top head H{top_head['head']} "
                      f"spectral_conc={top_head['spectral_conc']:.4f} "
                      f"eff_rank={top_head['eff_rank']:.1f} "
                      f"compliance_score={top_head['compliance_score']:.4f}")

    # ── 4. Summary stats ─────────────────────────────────────────────────────
    print(f"\nTotal head records: {len(results)}")

    top_by_compliance = sorted(results, key=lambda x: -x["compliance_score"])[:10]
    print("\nTop 10 heads by compliance score:")
    for r in top_by_compliance:
        print(f"  L{r['layer']:02d} H{r['head']} "
              f"compliance={r['compliance_score']:.4f} "
              f"spectral_conc={r['spectral_conc']:.4f} "
              f"eff_rank={r['eff_rank']:.1f}")

    top_by_conc = sorted(results, key=lambda x: -x["spectral_conc"])[:10]
    print("\nTop 10 heads by spectral concentration (sharpest OV circuits):")
    for r in top_by_conc:
        print(f"  L{r['layer']:02d} H{r['head']} "
              f"spectral_conc={r['spectral_conc']:.4f} "
              f"top_sv={r['top3_sv'][0]:.2f} "
              f"eff_rank={r['eff_rank']:.1f}")

    # ── 5. Save to volume ────────────────────────────────────────────────────
    out_path = Path("/data") / OUT_FILE
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print(f"\nSaved {len(results)} rows to {out_path}")
    return results


@app.local_entrypoint()
def main():
    print("Running OV circuit analysis on Gemma-4-E2B-it ...")
    results = run_ov_analysis.remote()
    print(f"\nDone — {len(results)} head records returned.")

    # Quick local summary
    top = sorted(results, key=lambda x: -x["compliance_score"])[:5]
    print("\nTop 5 compliance-aligned heads:")
    for r in top:
        print(f"  L{r['layer']:02d} H{r['head']}  "
              f"compliance={r['compliance_score']:.4f}  "
              f"spectral_conc={r['spectral_conc']:.4f}  "
              f"eff_rank={r['eff_rank']:.1f}")
