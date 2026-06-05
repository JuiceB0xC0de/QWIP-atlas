"""
extract_sae.py
--------------
PHASE 4 — SAE resolution pass for the Qwen3-8B atlas.

Re-expresses the atlas in Qwen-Scope SAE features (Alibaba's pretrained
residual-stream SAEs: 65,536 monosemantic features/layer, 36 layers). For every
SAE feature we compute, in ONE Modal job:

  * topic F-stat    — separation across the 16 behavior categories  (capability view)
  * bouncer F-stat  — separation on the corp-vs-authentic axis       (compliance view)
  * bouncer delta   — mean_corp − mean_auth  (+corp_leaning)
  * activation_rate — fraction of tokens the feature fires on

The high-bouncer / low-topic features are the surgical-gold steering knobs, and
each feature's decoder vector W_dec[:,i] is the knob itself.

EFFICIENCY (the encode is bandwidth-bound, not compute-bound — see README §SAE):
  * hook every layer's residual in a SINGLE forward pass per batch
  * keep only the sparse top-k; NEVER materialize the dense 65,536-wide vectors
  * accumulate per-(layer, feature, group) sums ONLINE → F-stat from sums alone
    (accumulators are ~8 MB/layer, so memory is a non-issue)

    modal run extract_sae.py --variant l0_50               # topic + bouncer, all layers
    modal run extract_sae.py --variant l0_100 --no-bouncer # topic only

Output (returned to the local entrypoint, written locally as npz per layer):
    sae_l{N}_{variant}.npz   with arrays: topic_fstat, bouncer_fstat,
                             bouncer_delta, mean_corp, mean_auth, activation_rate
Then:  python merge_sae.py --variant l0_50   (folds into atlas.sqlite)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config (mirrors extract_multi.py / extract_bouncer.py)
# ---------------------------------------------------------------------------
MODEL_ID    = "Qwen/Qwen3-8B-Base"
SAE_REPOS   = {
    "l0_50":  "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50",
    "l0_100": "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100",
}
TOPK        = {"l0_50": 50, "l0_100": 100}
D_SAE       = 65536
N_LAYERS    = 36                       # Qwen3-8B layers 0-35

CORPUS_REPO = "juiceb0xc0de/mapping-prompts"
CORPUS_FILE = "prompts.jsonl"
CORP_FILE   = "corporate_stems.jsonl"            # bouncer: corporate/compliance
AUTH_FILE   = "authentic_bella_samples.jsonl"    # bouncer: authentic

GPU_TYPE    = "A100-80GB"              # H100-80GB also great; bandwidth-bound either way
BATCH_SIZE  = 64
MAX_LEN     = 256                      # prompts are short; this is plenty
SAE_CACHE   = "/data/qwen-scope"       # SAEs cached on the volume across runs

app = modal.App("qwen-sae-resolution")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "gcc", "g++")
    .run_commands("python -m pip install --upgrade pip setuptools wheel")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("transformers>=5.5.0", "accelerate>=0.34.0", "numpy",
                 "huggingface_hub", "hf_transfer", "sentencepiece", "protobuf", "tqdm")
    .env({"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
          "HF_XET_HIGH_PERFORMANCE": "1"})
)

def _download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID, token=os.environ.get("HF_TOKEN"))

image = image.run_function(_download_model, secrets=[modal.Secret.from_name("huggingface")])
volume = modal.Volume.from_name("training_data", create_if_missing=True)


# ---------------------------------------------------------------------------
# Layer resolution (same as extract_multi.py)
# ---------------------------------------------------------------------------
def _resolve_layers(model):
    import torch.nn as nn
    from collections import deque
    queue = deque([model])
    while queue:
        m = queue.popleft()
        layers = getattr(m, "layers", None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return list(layers)
        for _, child in m.named_children():
            queue.append(child)
    raise RuntimeError("Cannot find layers ModuleList")


def _load_saes(variant: str, layers: list[int], device, dtype):
    """Download (cached on volume) and load one SAE per layer. Returns dict[L] -> (W_enc, b_enc)."""
    import torch
    from huggingface_hub import hf_hub_download
    repo = SAE_REPOS[variant]
    os.makedirs(SAE_CACHE, exist_ok=True)
    saes = {}
    for L in layers:
        fname = f"layer{L}.sae.pt"
        local = hf_hub_download(repo_id=repo, filename=fname, local_dir=f"{SAE_CACHE}/{variant}",
                                token=os.environ.get("HF_TOKEN"))
        sd = torch.load(local, map_location="cpu", weights_only=True)
        W_enc = sd["W_enc"].to(device=device, dtype=dtype)   # (65536, 4096)
        b_enc = sd["b_enc"].to(device=device, dtype=dtype)   # (65536,)
        saes[L] = (W_enc, b_enc)
        print(f"[sae] loaded {variant} layer {L:>2}  W_enc={tuple(W_enc.shape)}", flush=True)
    volume.commit()
    return saes


# ---------------------------------------------------------------------------
# Core: encode a corpus, accumulate per-(layer, feature, group) sums ONLINE
# ---------------------------------------------------------------------------
def _accumulate(corpus, group_of, n_groups, model, tokenizer, layers, saes,
                topk, batch_size, device):
    """One forward pass per batch; hook every layer's residual; encode through the
    SAE; scatter-add topk values into per-group running sums. Returns, per layer:
        sum[g, F], sumsq[g, F], n[g] (scalar tokens), active[F]
    all as torch tensors on `device`. No dense (.., 65536) vector is ever stored."""
    import torch
    from tqdm import tqdm

    target = list(saes.keys())
    acc = {L: {
        "sum":    torch.zeros(n_groups, D_SAE, dtype=torch.float64, device=device),
        "sumsq":  torch.zeros(n_groups, D_SAE, dtype=torch.float64, device=device),
        "active": torch.zeros(D_SAE, dtype=torch.float64, device=device),
    } for L in target}
    group_n = torch.zeros(n_groups, dtype=torch.float64, device=device)

    n_batches = (len(corpus) + batch_size - 1) // batch_size
    for bstart in tqdm(range(0, len(corpus), batch_size), total=n_batches, desc="batches", unit="batch"):
        batch = corpus[bstart: bstart + batch_size]
        prompts = [r["prompt"] for r in batch]
        groups  = [group_of(r) for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
        mask = enc["attention_mask"].bool()
        enc = {k: v.to(device) for k, v in enc.items()}

        captured: dict[int, "torch.Tensor"] = {}
        def mk(L):
            def _h(mod, inp, out):
                captured[L] = (out[0] if isinstance(out, tuple) else out).detach()
            return _h
        handles = [layers[L].register_forward_hook(mk(L)) for L in target]
        with torch.no_grad():
            model(**enc, use_cache=False)
        for h in handles: h.remove()

        # gather real (non-pad) token rows and their group id, per prompt
        for L in target:
            resid = captured[L]                                  # (B, T, 4096)
            W_enc, b_enc = saes[L]
            for gid in set(groups):
                rows_idx = [i for i, g in enumerate(groups) if g == gid]
                if not rows_idx: continue
                sub = resid[rows_idx]                            # (b, T, 4096)
                sub_mask = mask[rows_idx]                        # (b, T)
                tok = sub[sub_mask]                              # (n_real, 4096)
                if tok.numel() == 0: continue
                pre = tok.to(W_enc.dtype) @ W_enc.T + b_enc      # (n_real, 65536)
                vals, idx = pre.topk(topk, dim=-1)               # sparse top-k only
                vals = vals.clamp_min(0).double()                # ReLU-style; ignore negatives
                flat_i = idx.reshape(-1)
                flat_v = vals.reshape(-1)
                acc[L]["sum"][gid].index_add_(0, flat_i, flat_v)
                acc[L]["sumsq"][gid].index_add_(0, flat_i, flat_v * flat_v)
                acc[L]["active"].index_add_(0, flat_i, torch.ones_like(flat_v))
                if L == target[0]:
                    group_n[gid] += tok.shape[0]
            del resid
        captured.clear()
    return acc, group_n


def _fstat_from_sums(acc_L, group_n):
    """ANOVA F-stat per feature from per-group sums (computed entirely from sums)."""
    import torch
    s, ss = acc_L["sum"], acc_L["sumsq"]                 # (G, F)
    n = group_n.unsqueeze(1).clamp_min(1)                # (G,1)
    G = s.shape[0]
    N = group_n.sum().clamp_min(1)
    grand = s.sum(0) / N                                 # (F,)
    mean_g = s / n                                       # (F,) per group
    ssb = (group_n.unsqueeze(1) * (mean_g - grand) ** 2).sum(0)
    ssw = (ss - s ** 2 / n).sum(0).clamp_min(0)
    df_b, df_w = max(G - 1, 1), torch.clamp(N - G, min=1)
    F = (ssb / df_b) / (ssw / df_w + 1e-12)
    return F.float().cpu().numpy()                       # (65536,)


# ---------------------------------------------------------------------------
# Modal remote
# ---------------------------------------------------------------------------
@app.function(image=image, gpu=GPU_TYPE, timeout=86400,
              volumes={"/data": volume}, secrets=[modal.Secret.from_name("huggingface")])
def encode_remote(variant: str, layers: list[int], do_bouncer: bool, batch_size: int):
    import torch, numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download
    token = os.environ.get("HF_TOKEN")

    def _load_jsonl(repo, fname):
        p = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset", token=token)
        return [json.loads(l) for l in open(p) if l.strip()]

    print(f"[sae] corpus {CORPUS_REPO}/{CORPUS_FILE}")
    topic = _load_jsonl(CORPUS_REPO, CORPUS_FILE)
    cats  = sorted({r["category"] for r in topic})
    cat_id = {c: i for i, c in enumerate(cats)}
    print(f"[sae] {len(topic)} topic prompts across {len(cats)} categories")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=token)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, token=token, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    layers_mod = _resolve_layers(model)
    device = next(model.parameters()).device

    saes = _load_saes(variant, layers, device, torch.bfloat16)

    # ── topic pass ──
    print("[sae] === topic pass ===")
    t_acc, t_n = _accumulate(topic, lambda r: cat_id[r["category"]], len(cats),
                             model, tok, layers_mod, saes, TOPK[variant], batch_size, device)

    # ── bouncer pass (optional) ──
    # NOTE: corp/auth live locally at ~/sub-zero/corpora/ — they must be uploaded to
    # the mapping-prompts HF dataset for this pass to run on Modal. If they're not
    # there yet we skip bouncer gracefully so the (expensive) topic pass is never lost.
    b_acc = b_n = None
    corp = auth = None
    if do_bouncer:
        try:
            corp = _load_jsonl(CORPUS_REPO, CORP_FILE)
            auth = _load_jsonl(CORPUS_REPO, AUTH_FILE)
        except Exception as e:
            print(f"[sae] [warn] bouncer corpus not in {CORPUS_REPO} ({type(e).__name__}); "
                  f"skipping bouncer pass. Upload {CORP_FILE} + {AUTH_FILE} there to enable it.")
            corp = auth = None
    if corp is not None and auth is not None:
        print("[sae] === bouncer pass (corp vs auth) ===")
        def _norm(rows, lab):
            return [{"prompt": (r.get("prompt") or r.get("text") or r.get("stem") or ""), "_g": lab} for r in rows]
        bcorpus = _norm(corp, 0) + _norm(auth, 1)
        b_acc, b_n = _accumulate(bcorpus, lambda r: r["_g"], 2,
                                 model, tok, layers_mod, saes, TOPK[variant], batch_size, device)

    # ── reduce to per-feature scores per layer ──
    out: dict[int, dict] = {}
    N_topic = float(t_n.sum().item())
    for L in layers:
        rec = {
            "topic_fstat":     _fstat_from_sums(t_acc[L], t_n),
            "activation_rate": (t_acc[L]["active"] / max(N_topic, 1)).float().cpu().numpy(),
        }
        if b_acc is not None:
            rec["bouncer_fstat"] = _fstat_from_sums(b_acc[L], b_n)
            s, n = b_acc[L]["sum"], b_n.clamp_min(1).unsqueeze(1)
            mean = (s / n).float().cpu().numpy()                # (2, F)
            rec["mean_corp"], rec["mean_auth"] = mean[0], mean[1]
            rec["bouncer_delta"] = mean[0] - mean[1]
        out[L] = {k: v.tolist() for k, v in rec.items()}        # json-safe for return
        print(f"[sae] layer {L:>2} scored", flush=True)
    return {"variant": variant, "categories": cats, "n_topic_tokens": N_topic, "layers": out}


@app.local_entrypoint()
def main(variant: str = "l0_50", layers: str = "0-35",
         no_bouncer: bool = False, batch_size: int = BATCH_SIZE):
    import numpy as np
    layer_list = []
    for chunk in layers.split(","):
        if "-" in chunk:
            a, b = chunk.split("-"); layer_list += list(range(int(a), int(b) + 1))
        elif chunk.strip(): layer_list.append(int(chunk))
    layer_list = sorted(set(layer_list))
    assert variant in SAE_REPOS, f"variant must be one of {list(SAE_REPOS)}"

    print(f"[local] SAE encode  variant={variant}  layers={layer_list}  bouncer={not no_bouncer}")
    res = encode_remote.remote(variant, layer_list, not no_bouncer, batch_size)

    for L, rec in res["layers"].items():
        arrs = {k: np.asarray(v, dtype=np.float32) for k, v in rec.items()}
        np.savez_compressed(f"sae_l{L}_{variant}.npz", categories=np.array(res["categories"]), **arrs)
    print(f"\n[local] wrote sae_l*_{variant}.npz for {len(res['layers'])} layers")
    print(f"[local] next:  python merge_sae.py --variant {variant}")


if __name__ == "__main__":
    print("Run with:  modal run extract_sae.py --variant l0_50")
    raise SystemExit(0)
