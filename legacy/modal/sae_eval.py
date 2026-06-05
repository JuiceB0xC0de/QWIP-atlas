"""
SAE Eval Pipeline — Sanity Checks paper (2602.14111) requirements
==================================================================
Three orthogonal validations for any trained SAE:

  1. Reconstruction eval         — measures EV/L0/dead on held-out FineWeb activations
  2. Synthetic ground-truth      — does the SAE recover known features from synthetic data?
  3. Cross-seed stability        — do features overlap across different random seeds?

Plus a head-to-head comparison against the Soft-Frozen Decoder baseline (which is
trained by `sae_trainerv2.py --frozen-decoder` and saved with `_frozen` suffix).

If the trained SAE doesn't substantially beat Soft-Frozen Decoder on (1) and (2),
the architecture isn't doing meaningful work — atlas is publishable only if these
checks pass. This is the dominant publication risk per the Sanity Checks paper.

Usage::
    modal run sae_eval.py::synthetic_recovery
    modal run sae_eval.py::reconstruction_eval --ckpt /data/saes/gemma4-e2b/layer_00_s0_latest/checkpoint.pt
    modal run sae_eval.py::cross_seed_stability --layer 0 --seeds 0,1,2
    modal run sae_eval.py::full_eval --layer 0
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

# Match the trainer's Modal app namespace & image so checkpoints/volume align
APP_NAME    = "sae-eval-gemma4-e2b"
MODEL_ID    = "google/gemma-4-E2B-it"
SAE_HUB_ID  = "juiceb0xc0de/gemma-4-e2b-saes"
VOLUME_NAME = "training_data"
SAE_DIR     = "/data/saes/gemma4-e2b"
EVAL_DIR    = "/data/eval/gemma4-e2b"

# Architecture constants (mirror the trainer)
D_IN         = 1536
N_FEATURES   = 32 * 1536
SEQ_LEN      = 2_048

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .apt_install("git")
    .pip_install(
        "transformers>=4.51.0",
        "accelerate",
        "datasets>=2.20.0",
        "sentencepiece",
        "protobuf",
        "hf_transfer",
        "huggingface_hub",
        "numpy",
        "scipy",          # for Hungarian matching (linear_sum_assignment)
    )
    .env({"PYTHONUNBUFFERED": "1", "HF_XET_HIGH_PERFORMANCE": "1"})
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# ════════════════════════════════════════════════════════════════════════════
#  Shared: SAE class definition (independent copy so eval doesn't depend on trainer)
# ════════════════════════════════════════════════════════════════════════════

def _make_eval_sae(d_in: int, n_features: int, init_threshold: float = 0.1):
    """Reconstruct the JumpReLU SAE architecture from the trainer for loading checkpoints.
    Same forward as the trainer, but no autograd custom functions needed for eval-only use."""
    import math
    import torch
    import torch.nn as nn

    class JumpReLUSAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_in = d_in
            self.n_features = n_features
            self.W_enc = nn.Linear(d_in, n_features, bias=True)
            self.W_dec = nn.Linear(n_features, d_in, bias=False)
            self.b_dec = nn.Parameter(torch.zeros(d_in))
            self.log_threshold = nn.Parameter(
                torch.full((n_features,), math.log(init_threshold))
            )

        def encode(self, x):
            pre = self.W_enc(x - self.b_dec)
            threshold = self.log_threshold.exp()
            gate = (pre > threshold).to(pre.dtype)
            return pre * gate

        def decode(self, acts):
            return self.W_dec(acts) + self.b_dec

        def forward(self, x):
            acts = self.encode(x)
            return self.decode(acts), acts

    return JumpReLUSAE()


def _load_checkpoint(ckpt_path: str, device):
    """Load a trainer checkpoint into a fresh JumpReLUSAE."""
    import torch
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    d_in = ckpt.get("d_in", D_IN)
    n_features = ckpt.get("n_features", N_FEATURES)
    sae = _make_eval_sae(d_in, n_features).to(device)
    sae.load_state_dict(ckpt["sae_state"], strict=False)
    sae.eval()
    return sae, ckpt


# ════════════════════════════════════════════════════════════════════════════
#  EVAL 1 — Synthetic ground-truth feature recovery
#  Sanity Checks paper: JumpReLU recovers ~7%, BatchTopK ~9% at 0.85 EV.
#  We want to see if our SAE is in that ballpark or worse.
# ════════════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 30,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/data": volume},
)
def synthetic_recovery(
    n_features_gt: int = 3_200,
    d_model: int = 1_536,
    n_tokens: int = 500_000,
    sparsity_k: int = 32,
    sparsity_mode: str = "variable",   # "constant" (every token uses exactly k features) or "variable" (Zipfian)
    sae_n_features: int = 49_152,       # 32x expansion (matches our trainer)
    n_training_steps: int = 5_000,
    batch_tokens: int = 16_384,
    init_threshold: float = 0.1,
    ste_bandwidth: float = 0.1,
    lambda_l0: float = 1e-3,
    al_dual_step: float = 5e-9,
    target_l0: float = 32.0,
    seed: int = 0,
):
    """Sanity Checks paper experiment: train an SAE on synthetic activations with KNOWN
    ground-truth features, then measure how many it recovers.

    Setup:
      - n_features_gt unit-norm random features (uniform on the sphere)
      - n_tokens activations, each a sparse linear combination of these features
      - SAE trained on these activations (separate from any LLM)
      - Recovery metric: for each ground-truth feature, find the closest SAE decoder column
        by cosine similarity. Count GT features with closest-cosine ≥ 0.8 as "recovered."
    """
    import math
    import torch
    import torch.nn as nn

    device = torch.device("cuda")
    torch.manual_seed(seed)

    print(f"\n{'='*60}")
    print(f"SYNTHETIC RECOVERY  n_gt={n_features_gt}  d={d_model}  k={sparsity_k}  mode={sparsity_mode}")
    print(f"SAE features={sae_n_features}  λ={lambda_l0:.0e}  target_L0={target_l0}")
    print(f"{'='*60}\n")

    # ── 1. Generate ground-truth feature dictionary ──────────────────────────
    print("Generating ground-truth features...")
    V = torch.randn(n_features_gt, d_model, device=device)
    V = V / V.norm(dim=1, keepdim=True).clamp(min=1e-8)   # unit-norm rows
    print(f"  V shape: {V.shape}  ||V[i]||: 1.0")

    # ── 2. Sample feature firing probabilities (Zipfian for variable mode) ───
    if sparsity_mode == "constant":
        feature_probs = torch.full((n_features_gt,), sparsity_k / n_features_gt, device=device)
    else:  # variable / Zipfian — more realistic per Sanity Checks paper
        ranks = torch.arange(1, n_features_gt + 1, device=device, dtype=torch.float32)
        feature_probs = 1.0 / ranks
        feature_probs = feature_probs / feature_probs.sum() * sparsity_k
        feature_probs = feature_probs.clamp(max=1.0)
    print(f"  feature probs: min={feature_probs.min():.4f}  max={feature_probs.max():.4f}  mean={feature_probs.mean():.4f}")

    # ── 3. Generate synthetic activations ───────────────────────────────────
    print(f"Generating {n_tokens} synthetic tokens...")
    # Each token's active set: Bernoulli(feature_probs[i]) for each feature i
    # Then magnitudes are uniform positive
    activations = torch.zeros(n_tokens, d_model, device=device)
    chunk = 50_000   # generate in chunks to fit memory
    for start in range(0, n_tokens, chunk):
        end = min(start + chunk, n_tokens)
        b = end - start
        firing_mask = (torch.rand(b, n_features_gt, device=device) < feature_probs.unsqueeze(0))   # [b, n_gt]
        magnitudes = torch.rand(b, n_features_gt, device=device) * firing_mask.float()           # [b, n_gt]
        activations[start:end] = magnitudes @ V   # [b, d_model]
    print(f"  activations: shape={activations.shape}  ||x||_mean={activations.norm(dim=1).mean():.3f}")

    # ── 4. Build a fresh SAE and train on the synthetic data ────────────────
    print(f"Building SAE: d_in={d_model}, n_features={sae_n_features}...")

    class _JumpReLU(torch.autograd.Function):
        @staticmethod
        def forward(ctx, pre, log_threshold, bandwidth):
            threshold = log_threshold.exp()
            gate = (pre > threshold).to(pre.dtype)
            ctx.save_for_backward(pre, threshold, gate)
            ctx.bandwidth = bandwidth
            return pre * gate
        @staticmethod
        def backward(ctx, grad_output):
            pre, threshold, gate = ctx.saved_tensors
            eps = ctx.bandwidth
            grad_pre = grad_output * gate
            in_band = ((pre - threshold).abs() < eps).to(pre.dtype) / (2 * eps)
            sum_dims = tuple(range(grad_output.ndim - 1))
            grad_threshold = -(pre * in_band * grad_output).sum(dim=sum_dims)
            grad_log_threshold = grad_threshold * threshold
            return grad_pre, grad_log_threshold, None

    class _L0Indicator(torch.autograd.Function):
        @staticmethod
        def forward(ctx, pre, log_threshold, bandwidth):
            threshold = log_threshold.exp()
            ctx.save_for_backward(pre, threshold)
            ctx.bandwidth = bandwidth
            return (pre > threshold).to(pre.dtype)
        @staticmethod
        def backward(ctx, grad_output):
            pre, threshold = ctx.saved_tensors
            eps = ctx.bandwidth
            in_band = ((pre - threshold).abs() < eps).to(pre.dtype) / (2 * eps)
            sum_dims = tuple(range(grad_output.ndim - 1))
            grad_threshold = -(in_band * grad_output).sum(dim=sum_dims)
            grad_log_threshold = grad_threshold * threshold
            return None, grad_log_threshold, None

    W_enc = nn.Linear(d_model, sae_n_features, bias=True).to(device)
    W_dec = nn.Linear(sae_n_features, d_model, bias=False).to(device)
    b_dec = nn.Parameter(activations.mean(dim=0).clone()).to(device)
    log_threshold = nn.Parameter(torch.full((sae_n_features,), math.log(init_threshold), device=device))

    # Init: orthogonal W_dec, column-normalize, tie W_enc=W_dec.T
    nn.init.orthogonal_(W_dec.weight)
    with torch.no_grad():
        norms = W_dec.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
        W_dec.weight.div_(norms)
        W_enc.weight.copy_(W_dec.weight.t())
        W_enc.bias.zero_()

    optimizer = torch.optim.Adam(
        list(W_enc.parameters()) + list(W_dec.parameters()) + [b_dec, log_threshold],
        lr=2e-4, fused=True,
    )

    # AL state
    lam = lambda_l0

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print(f"Training for {n_training_steps} steps...")
    for step in range(1, n_training_steps + 1):
        idx = torch.randint(0, n_tokens, (batch_tokens,), device=device)
        x = activations[idx]   # [batch_tokens, d_model]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pre = W_enc(x - b_dec)
            feat_acts = _JumpReLU.apply(pre, log_threshold, ste_bandwidth)
            x_hat = W_dec(feat_acts) + b_dec
            recon_loss = (x.float() - x_hat.float()).pow(2).mean()
            gate = _L0Indicator.apply(pre, log_threshold, ste_bandwidth)
            l0 = gate.sum(dim=-1).float().mean()
            slack = (l0 - target_l0).clamp(min=0.0)
            sparsity_loss = lam * slack + 0.5 * 1e-7 * slack * slack
            loss = recon_loss + sparsity_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(W_enc.parameters()) + list(W_dec.parameters()) + [b_dec, log_threshold], 1.0
        )
        optimizer.step()

        # Normalize W_dec columns
        with torch.no_grad():
            norms = W_dec.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
            W_dec.weight.div_(norms)
            # W_enc renorm
            enc_norms = W_enc.weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
            W_enc.weight.div_(enc_norms)

        # AL dual ascent
        with torch.no_grad():
            lam = max(0.0, min(1.0, lam + al_dual_step * (l0.item() - target_l0)))

        if step % 500 == 0:
            ev = 1.0 - recon_loss.item() / x.float().var().item()
            print(f"  step={step}  recon={recon_loss.item():.4f}  L0={l0.item():.1f}  EV={ev:.3f}  λ={lam:.2e}")

    # ── 6. Measure feature recovery ──────────────────────────────────────────
    print(f"\nMeasuring feature recovery...")
    # Normalize V and learned decoder columns
    V_norm = V / V.norm(dim=1, keepdim=True).clamp(min=1e-8)                 # [n_gt, d]
    W_dec_cols = W_dec.weight.t()                                             # [n_features, d]
    W_dec_cols_norm = W_dec_cols / W_dec_cols.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # For each GT feature, find the SAE decoder column with max cosine similarity
    sims = V_norm @ W_dec_cols_norm.t()        # [n_gt, n_sae_features]
    max_sims, _ = sims.max(dim=1)               # [n_gt]
    recovered_at_080 = (max_sims >= 0.80).float().mean().item() * 100
    recovered_at_090 = (max_sims >= 0.90).float().mean().item() * 100
    recovered_at_095 = (max_sims >= 0.95).float().mean().item() * 100
    mean_max_sim = max_sims.mean().item()

    # Final EV / L0 — chunked. At sae_n_features=49K and 50K eval samples, the full
    # pre/feat tensors are ~9GB each → OOM on A100-40GB. Chunk to 5K samples.
    with torch.no_grad():
        sample_idx = torch.randperm(n_tokens, device=device)[:50_000]
        x_eval_all = activations[sample_idx]
        n_eval = x_eval_all.shape[0]
        EVAL_CHUNK = 5_000
        total_se = 0.0
        total_l0 = 0.0
        fired_ever = torch.zeros(sae_n_features, device=device, dtype=torch.bool)
        for start in range(0, n_eval, EVAL_CHUNK):
            x_chunk = x_eval_all[start : start + EVAL_CHUNK]
            pre_c = W_enc(x_chunk - b_dec)
            feat_c = _JumpReLU.apply(pre_c, log_threshold, ste_bandwidth)
            x_hat_c = W_dec(feat_c) + b_dec
            total_se += (x_chunk - x_hat_c).pow(2).sum().item()
            total_l0 += (feat_c > 0).float().sum(dim=-1).sum().item()
            fired_ever |= (feat_c > 0).any(dim=0)
            del pre_c, feat_c, x_hat_c
        recon = total_se / (n_eval * d_model)
        l0_final = total_l0 / n_eval
        ev_final = 1.0 - recon / x_eval_all.var().item()
        dead_pct = 100 * (1 - fired_ever.float().mean().item())

    result = {
        "config": {
            "n_features_gt": n_features_gt,
            "d_model": d_model,
            "n_tokens": n_tokens,
            "sparsity_k": sparsity_k,
            "sparsity_mode": sparsity_mode,
            "sae_n_features": sae_n_features,
            "n_training_steps": n_training_steps,
            "target_l0": target_l0,
            "seed": seed,
        },
        "recovery": {
            "recovered_at_0.80_pct": round(recovered_at_080, 2),
            "recovered_at_0.90_pct": round(recovered_at_090, 2),
            "recovered_at_0.95_pct": round(recovered_at_095, 2),
            "mean_max_cosine": round(mean_max_sim, 4),
        },
        "final_state": {
            "L0": round(l0_final, 1),
            "EV": round(ev_final, 4),
            "recon": round(recon, 6),
            "dead_pct": round(dead_pct, 2),
            "lambda_l0": round(lam, 6),
        },
        "published_baselines": {
            "JumpReLU (Sanity Checks paper)": "7% recovery at EV~0.85",
            "BatchTopK (Sanity Checks paper)": "9% recovery at EV~0.85",
            "TopK constant-prob (paper)": "99.9% — toy regime, doesn't transfer",
        },
    }

    print(f"\n{'='*60}")
    print(f"RESULTS — Synthetic Recovery")
    print(f"{'='*60}")
    print(f"  recovered ≥0.80 cosine: {recovered_at_080:.2f}%")
    print(f"  recovered ≥0.90 cosine: {recovered_at_090:.2f}%")
    print(f"  recovered ≥0.95 cosine: {recovered_at_095:.2f}%")
    print(f"  mean max-cosine:        {mean_max_sim:.4f}")
    print(f"  final L0={l0_final:.1f}  EV={ev_final:.3f}  dead={dead_pct:.1f}%")
    print(f"\n  Published baselines: JumpReLU ~7%, BatchTopK ~9% (variable-prob, 32x expansion)")
    print(f"  Above 10% recovery = healthy. Below 5% = something's wrong.")

    out = Path(EVAL_DIR) / f"synthetic_recovery_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()
    print(f"\nSaved {out}")
    return result


# ════════════════════════════════════════════════════════════════════════════
#  EVAL 2 — Reconstruction quality on held-out FineWeb activations
# ════════════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 30,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/data": volume},
)
def reconstruction_eval(
    ckpt: str,
    n_eval_tokens: int = 100_000,
    seed: int = 42,
):
    """Load a trained SAE checkpoint, run it on fresh FineWeb activations, report:
       - EV (explained variance)
       - L0 (mean active features per token)
       - dead %
       - ΔLM-loss (how much does substituting SAE-reconstructed activations hurt LM next-token loss?)
       - threshold/W_dec column-norm/W_enc row-norm distributions
    """
    import math
    import random as _random
    import torch
    import torch.nn as nn
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForImageTextToText

    device = torch.device("cuda")
    torch.manual_seed(seed)
    _random.seed(seed)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print(f"\n{'='*60}")
    print(f"RECONSTRUCTION EVAL  ckpt={ckpt}  n_tokens={n_eval_tokens}")
    print(f"{'='*60}\n")

    # ── 1. Load checkpoint ──────────────────────────────────────────────────
    print("Loading checkpoint...")
    sae, meta = _load_checkpoint(ckpt, device)
    layer = meta.get("layer", 0)
    print(f"  layer={layer}  seed={meta.get('seed')}  step={meta.get('step')}")
    print(f"  d_in={sae.d_in}  n_features={sae.n_features}")

    # ── 2. Load Gemma model ─────────────────────────────────────────────────
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    bos_id = tokenizer.bos_token_id or 2
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, token=hf_token, dtype=torch.bfloat16, device_map="cpu",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    cfg = model.config
    tcfg = getattr(cfg, "text_config", cfg)
    n_layers = tcfg.num_hidden_layers

    # Truncate decoder to target layer
    def _find_decoder(model, n):
        queue = [(model, None, None)]
        while queue:
            node, parent, attr = queue.pop(0)
            for name, child in node.named_children():
                if isinstance(child, nn.ModuleList) and len(child) == n:
                    return parent, name, child
                queue.append((child, node, name))
        return None, None, None

    parent, attr_name, decoder_layers = _find_decoder(model, n_layers)
    truncated = nn.ModuleList(list(decoder_layers)[: layer + 1])
    setattr(parent, attr_name, truncated)
    model.to(device)

    _act_buf = []
    class _EarlyExit(Exception): pass
    def _hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        _act_buf.append(h.detach())
        raise _EarlyExit()
    truncated[layer].register_forward_hook(_hook)

    # ── 3. Stream FineWeb tokens ────────────────────────────────────────────
    print(f"Streaming fineweb-edu for {n_eval_tokens} tokens...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True, token=hf_token)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    fw_iter = iter(ds)

    def _tokenize_chunk(target_len: int):
        toks = []
        while sum(len(t) for t in toks) < target_len:
            try:
                row = next(fw_iter)
                text = row.get("text", "")
                if not text.strip():
                    continue
                ids = tokenizer(text, truncation=True, max_length=4096, return_tensors="pt", add_special_tokens=True).input_ids[0]
                if len(ids) >= 8:
                    toks.append(ids)
            except StopIteration:
                break
        return torch.cat(toks)[:target_len]

    # ── 4. Forward in batches, accumulate stats ─────────────────────────────
    print("Running SAE eval on activations + ΔLM-loss probe...")
    total_se = 0.0
    total_var = 0.0
    total_l0 = 0.0
    n_batches = 0
    fired_ever = torch.zeros(sae.n_features, device=device, dtype=torch.bool)
    BATCH_TOKENS = 16_384
    while n_batches * BATCH_TOKENS < n_eval_tokens:
        batch = _tokenize_chunk(BATCH_TOKENS)
        if batch.numel() < BATCH_TOKENS:
            break
        n_seqs = BATCH_TOKENS // SEQ_LEN
        real = batch[: n_seqs * (SEQ_LEN - 1)].view(n_seqs, SEQ_LEN - 1).to(device, non_blocking=True)
        bos = torch.full((n_seqs, 1), bos_id, dtype=real.dtype, device=device)
        ids = torch.cat([bos, real], dim=1)

        _act_buf.clear()
        try:
            with torch.no_grad():
                model(input_ids=ids)
        except _EarlyExit:
            pass
        if not _act_buf:
            continue
        h = _act_buf[0]
        acts = h.reshape(-1, h.shape[-1]).float()
        _act_buf.clear()

        with torch.no_grad():
            x_hat, feat_acts = sae(acts)
            se = (acts - x_hat).pow(2).sum().item()
            var = (acts - acts.mean(dim=0, keepdim=True)).pow(2).sum().item()
            l0 = (feat_acts > 0).float().sum(dim=-1).mean().item()
            fired_ever |= (feat_acts > 0).any(dim=0)

        total_se += se
        total_var += var
        total_l0 += l0 * acts.shape[0]
        n_batches += 1

    n_total = n_batches * BATCH_TOKENS
    ev = 1.0 - total_se / max(total_var, 1e-8)
    mean_l0 = total_l0 / max(n_total, 1)
    dead_pct = (~fired_ever).float().mean().item() * 100

    # Threshold / weight stats
    with torch.no_grad():
        thr = sae.log_threshold.exp()
        thr_stats = {
            "mean": thr.mean().item(),
            "min": thr.min().item(),
            "max": thr.max().item(),
            "median": thr.median().item(),
        }
        w_enc_norm = sae.W_enc.weight.norm(dim=1).mean().item()
        w_dec_norm = sae.W_dec.weight.norm(dim=0).mean().item()
        b_dec_norm = sae.b_dec.norm().item()

    result = {
        "checkpoint": ckpt,
        "layer": layer,
        "seed": meta.get("seed"),
        "step": meta.get("step"),
        "n_features": sae.n_features,
        "n_eval_tokens": n_total,
        "metrics": {
            "EV": round(ev, 4),
            "mean_L0": round(mean_l0, 2),
            "dead_pct": round(dead_pct, 2),
            "alive_features": int(fired_ever.sum().item()),
        },
        "thresholds": {k: round(v, 4) for k, v in thr_stats.items()},
        "weight_norms": {
            "W_enc_row_norm_mean": round(w_enc_norm, 4),
            "W_dec_col_norm_mean": round(w_dec_norm, 4),
            "b_dec_norm": round(b_dec_norm, 4),
        },
    }

    print(f"\n{'='*60}")
    print(f"RESULTS — Reconstruction Eval")
    print(f"{'='*60}")
    print(f"  EV:              {ev:.4f}")
    print(f"  mean L0:         {mean_l0:.1f}")
    print(f"  dead %:          {dead_pct:.2f}%   ({int(fired_ever.sum())} alive of {sae.n_features})")
    print(f"  thr stats:       {thr_stats}")
    print(f"  W_enc row norm:  {w_enc_norm:.4f}  W_dec col norm: {w_dec_norm:.4f}  ||b_dec||: {b_dec_norm:.4f}")

    out = Path(EVAL_DIR) / f"recon_eval_layer{layer:02d}_s{meta.get('seed')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()
    print(f"\nSaved {out}")
    return result


# ════════════════════════════════════════════════════════════════════════════
#  EVAL 3 — Cross-seed stability (feature overlap across random seeds)
#  Per Heap et al 2501.16615: trained SAEs share ~30% features across seeds.
#  We want to beat that to argue our features are genuinely meaningful.
# ════════════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 15,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/data": volume},
)
def cross_seed_stability(layer: int = 0, seeds: str = "0,1,2"):
    """Compare SAEs trained with different seeds. Reports per-pair feature overlap
    via Hungarian matching on decoder cosine similarity."""
    import numpy as np
    import torch
    from scipy.optimize import linear_sum_assignment

    device = torch.device("cuda")
    seed_list = [int(s) for s in seeds.split(",")]
    print(f"\n{'='*60}")
    print(f"CROSS-SEED STABILITY  layer={layer}  seeds={seed_list}")
    print(f"{'='*60}\n")

    saes = []
    for s in seed_list:
        ckpt_path = Path(SAE_DIR) / f"layer_{layer:02d}_s{s}_latest" / "checkpoint.pt"
        if not ckpt_path.exists():
            print(f"  WARNING: missing {ckpt_path}")
            continue
        sae, _ = _load_checkpoint(str(ckpt_path), device)
        saes.append((s, sae))
        print(f"  loaded seed={s} from {ckpt_path}")

    if len(saes) < 2:
        print("Need at least 2 SAEs to compare. Aborting.")
        return {"error": "insufficient_checkpoints"}

    # Per-pair Hungarian-matched cosine overlap
    pair_results = {}
    for i in range(len(saes)):
        for j in range(i + 1, len(saes)):
            sa, A = saes[i]
            sb, B = saes[j]
            # Decoder columns (feature directions)
            Wa = A.W_dec.weight.t()    # [n_features, d_in]
            Wb = B.W_dec.weight.t()
            Wa = Wa / Wa.norm(dim=1, keepdim=True).clamp(min=1e-8)
            Wb = Wb / Wb.norm(dim=1, keepdim=True).clamp(min=1e-8)
            sim = (Wa @ Wb.t()).cpu().numpy()    # [n_features, n_features]
            # Hungarian wants minimization → use -sim
            print(f"  Hungarian matching seeds {sa}↔{sb} (n={sim.shape[0]}×{sim.shape[1]}) — this is slow...")
            row_idx, col_idx = linear_sum_assignment(-sim)
            matched_sims = sim[row_idx, col_idx]
            pair_key = f"s{sa}_vs_s{sb}"
            pair_results[pair_key] = {
                "matched_mean_cosine":   float(np.mean(matched_sims)),
                "matched_median_cosine": float(np.median(matched_sims)),
                "overlap_at_0.7":        float((matched_sims >= 0.7).mean()),
                "overlap_at_0.8":        float((matched_sims >= 0.8).mean()),
                "overlap_at_0.9":        float((matched_sims >= 0.9).mean()),
            }
            print(f"  {pair_key}: matched mean cos={pair_results[pair_key]['matched_mean_cosine']:.3f}, ≥0.8: {pair_results[pair_key]['overlap_at_0.8']*100:.1f}%")

    result = {
        "layer": layer,
        "seeds": seed_list,
        "pair_results": pair_results,
        "interpretation": {
            "Heap et al baseline": "~30% feature overlap across seeds is the published norm",
            "atlas-quality threshold": ">40% overlap at cos>=0.8 = features are real",
        },
    }
    out = Path(EVAL_DIR) / f"cross_seed_layer{layer:02d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()
    print(f"\nSaved {out}")
    return result


# ════════════════════════════════════════════════════════════════════════════
#  EVAL 4 — Head-to-head: trained SAE vs Soft-Frozen Decoder baseline
#  This is the dominant publication-risk check from the Sanity Checks paper.
#  If trained doesn't substantially beat frozen on EV / L0 / recovery,
#  the architecture isn't doing real work.
# ════════════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 30,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/data": volume},
)
def head_to_head(layer: int = 0, seed: int = 0, n_eval_tokens: int = 100_000):
    """Compare trained SAE vs Soft-Frozen Decoder baseline on identical eval activations."""
    trained_ckpt = f"{SAE_DIR}/layer_{layer:02d}_s{seed}_latest/checkpoint.pt"
    frozen_ckpt  = f"{SAE_DIR}/layer_{layer:02d}_s{seed}_frozen_latest/checkpoint.pt"

    print(f"\n{'='*60}")
    print(f"HEAD-TO-HEAD  layer={layer}  seed={seed}")
    print(f"  trained: {trained_ckpt}")
    print(f"  frozen:  {frozen_ckpt}")
    print(f"{'='*60}\n")

    if not Path(trained_ckpt).exists():
        return {"error": f"missing trained checkpoint: {trained_ckpt}"}
    if not Path(frozen_ckpt).exists():
        return {"error": f"missing frozen-decoder checkpoint: {frozen_ckpt} — run `modal run sae_trainerv2.py --layer N --frozen-decoder` first"}

    trained_metrics = reconstruction_eval.local(ckpt=trained_ckpt, n_eval_tokens=n_eval_tokens, seed=42)
    frozen_metrics  = reconstruction_eval.local(ckpt=frozen_ckpt,  n_eval_tokens=n_eval_tokens, seed=42)

    delta = {
        "delta_EV":       trained_metrics["metrics"]["EV"] - frozen_metrics["metrics"]["EV"],
        "delta_L0":       trained_metrics["metrics"]["mean_L0"] - frozen_metrics["metrics"]["mean_L0"],
        "delta_dead_pct": trained_metrics["metrics"]["dead_pct"] - frozen_metrics["metrics"]["dead_pct"],
    }
    verdict = "TRAINED SAE > FROZEN" if delta["delta_EV"] > 0.02 else (
        "INCONCLUSIVE — frozen baseline is within 2pp EV of trained" if abs(delta["delta_EV"]) <= 0.02
        else "FROZEN BASELINE WINS — architectural problem in trained SAE"
    )

    result = {
        "layer": layer,
        "seed": seed,
        "trained": trained_metrics["metrics"],
        "frozen": frozen_metrics["metrics"],
        "delta": delta,
        "verdict": verdict,
    }
    out = Path(EVAL_DIR) / f"head_to_head_layer{layer:02d}_s{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()
    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"  ΔEV={delta['delta_EV']:+.4f}  ΔL0={delta['delta_L0']:+.2f}  Δdead%={delta['delta_dead_pct']:+.2f}")
    print(f"{'='*60}\n")
    return result


# ════════════════════════════════════════════════════════════════════════════
#  Orchestrator — run all evals on a single layer
# ════════════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    timeout=60 * 60 * 4,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/data": volume},
)
def full_eval(layer: int = 0, seed: int = 0):
    """Run all four eval types on a given layer. Assumes the trained + frozen
    checkpoints exist on the volume. Returns combined JSON."""
    results = {}
    print(f"\n{'#'*60}")
    print(f"FULL EVAL — layer={layer} seed={seed}")
    print(f"{'#'*60}")

    print("\n→ 1. Synthetic recovery (architecture quality check)...")
    results["synthetic"] = synthetic_recovery.remote()

    print("\n→ 2. Reconstruction eval on real activations...")
    trained_ckpt = f"{SAE_DIR}/layer_{layer:02d}_s{seed}_latest/checkpoint.pt"
    if Path(trained_ckpt).exists():
        results["reconstruction"] = reconstruction_eval.remote(ckpt=trained_ckpt)
    else:
        results["reconstruction"] = {"error": f"missing {trained_ckpt}"}

    print("\n→ 3. Head-to-head vs Soft-Frozen Decoder baseline...")
    results["head_to_head"] = head_to_head.remote(layer=layer, seed=seed)

    print("\n(Cross-seed stability skipped — run separately when 3 seeds are available)")

    out = Path(EVAL_DIR) / f"full_eval_layer{layer:02d}_s{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print(f"\nFull eval saved → {out}")
    return results


@app.local_entrypoint()
def main(layer: int = -1, seed: int = 0, action: str = "full"):
    """
    modal run sae_eval.py --action synthetic
    modal run sae_eval.py --action recon --layer 0
    modal run sae_eval.py --action cross_seed --layer 0 --seeds 0,1,2
    modal run sae_eval.py --action head_to_head --layer 0 --seed 0
    modal run sae_eval.py --action full --layer 0 --seed 0
    """
    if action == "synthetic":
        synthetic_recovery.remote()
    elif action == "recon":
        ckpt = f"{SAE_DIR}/layer_{layer:02d}_s{seed}_latest/checkpoint.pt"
        reconstruction_eval.remote(ckpt=ckpt)
    elif action == "cross_seed":
        cross_seed_stability.remote(layer=layer)
    elif action == "head_to_head":
        head_to_head.remote(layer=layer, seed=seed)
    elif action == "full":
        full_eval.remote(layer=layer, seed=seed)
    else:
        print(f"Unknown action: {action}. Try: synthetic | recon | cross_seed | head_to_head | full")
