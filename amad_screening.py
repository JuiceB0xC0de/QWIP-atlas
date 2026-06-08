"""
Behavioral Geometry Pre-Screening for SAE Budget Allocation
============================================================
Uses per-neuron F-stat separation vectors (separation.npy) as the layer
similarity metric — measures how similarly each layer discriminates behavioral
categories. Layers with similar discrimination geometry can share one SAE.

This is our metric: behavioral separability angular distance, not raw activation
geometry. The F-stats encode exactly what we care about for SAE budget decisions.

Output: layer_groups.json + screening report printed to stdout.
"""

import json
import numpy as np
from pathlib import Path

ATLAS_DIR  = Path("atlas/layers")
COMPONENTS = ["gate", "up", "mlp"]

# Angular distance threshold for grouping (degrees).
# Tune this based on the distribution we actually see.
GROUP_THRESHOLD_DEG = 25.0

TIER_HIGH   = 12.0   # F-stat peak → full SAE, don't group
TIER_MEDIUM = 3.0    # F-stat peak → group-SAE candidate
# below TIER_MEDIUM → skip


def load_fstat_vec(layer: int, comp: str):
    p = ATLAS_DIR / str(layer) / "components" / comp / "separation.npy"
    if not p.exists():
        return None
    v = np.load(p).astype(np.float64)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else None


def angular_dist_deg(a, b):
    # Truncate to min dim (handles 6144 vs 12288 boundary at L15)
    n = min(len(a), len(b))
    a, b = a[:n].copy(), b[:n].copy()
    a /= np.linalg.norm(a) + 1e-10
    b /= np.linalg.norm(b) + 1e-10
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def fstat_peak(layer: int):
    peak = 0.0
    for comp in COMPONENTS:
        p = ATLAS_DIR / str(layer) / "components" / comp / "separation.npy"
        if p.exists():
            peak = max(peak, float(np.load(p).max()))
    return peak


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", default="atlas", help="Atlas root containing layers/<N>/components")
    args = parser.parse_args()

    atlas_layers = Path(args.atlas) / "layers"
    if not atlas_layers.exists():
        raise SystemExit(f"atlas layers dir not found: {atlas_layers}")

    global ATLAS_DIR
    ATLAS_DIR = atlas_layers
    n_layers = max(int(p.name) for p in ATLAS_DIR.iterdir() if p.is_dir()) + 1
    print(f"Behavioral geometry screening — {n_layers} layers, components: {COMPONENTS}")

    # ── 1. Load F-stat vectors ───────────────────────────────────────────────
    vecs = {}
    for layer in range(n_layers):
        for comp in COMPONENTS:
            v = load_fstat_vec(layer, comp)
            if v is not None:
                vecs[(layer, comp)] = v

    # ── 2. Per-component angular distance matrix ─────────────────────────────
    dist_per_comp = {}
    for comp in COMPONENTS:
        mat = np.full((n_layers, n_layers), np.nan)
        for i in range(n_layers):
            vi = vecs.get((i, comp))
            if vi is None:
                continue
            for j in range(n_layers):
                vj = vecs.get((j, comp))
                if vj is None:
                    continue
                mat[i, j] = angular_dist_deg(vi, vj)
        dist_per_comp[comp] = mat

    combined = np.nanmean(np.stack(list(dist_per_comp.values())), axis=0)

    # ── 3. F-stat peaks per layer ────────────────────────────────────────────
    peaks = {l: fstat_peak(l) for l in range(n_layers)}

    # ── 4. Greedy contiguous grouping ────────────────────────────────────────
    def greedy_group(layers, threshold):
        groups, current = [], [layers[0]]
        for l in layers[1:]:
            max_dist = max(
                combined[l][j] if not np.isnan(combined[l][j]) else 999
                for j in current
            )
            if max_dist < threshold:
                current.append(l)
            else:
                groups.append(current)
                current = [l]
        groups.append(current)
        return groups

    present = [l for l in range(n_layers) if any((l, c) in vecs for c in COMPONENTS)]
    raw_groups = greedy_group(present, GROUP_THRESHOLD_DEG)

    # ── 5. Assign tiers — split high-signal groups into per-layer runs ───────
    layer_groups = []
    total_runs   = 0
    for g in raw_groups:
        group_peak = max(peaks[l] for l in g)
        if group_peak >= TIER_HIGH:
            for l in g:
                lp   = peaks[l]
                tier = "high" if lp >= TIER_HIGH else ("medium" if lp >= TIER_MEDIUM else "skip")
                runs = 0 if tier == "skip" else 1
                layer_groups.append({"layers": [l], "tier": tier,
                                     "peak_fstat": round(lp, 2), "sae_runs": runs})
                total_runs += runs
        elif group_peak >= TIER_MEDIUM:
            layer_groups.append({"layers": g, "tier": "medium",
                                  "peak_fstat": round(group_peak, 2), "sae_runs": 1})
            total_runs += 1
        else:
            layer_groups.append({"layers": g, "tier": "skip",
                                  "peak_fstat": round(group_peak, 2), "sae_runs": 0})

    # ── 6. Report ─────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SAE BUDGET — BEHAVIORAL GEOMETRY SCREENING")
    print(f"{'='*65}")
    print(f"Threshold: {GROUP_THRESHOLD_DEG}°  |  Groups: {len(layer_groups)}  "
          f"|  SAE runs: {total_runs} / {n_layers}")
    print()
    print(f"{'#':<4} {'Layers':<38} {'Tier':<8} {'F-peak':>7} {'Runs':>5}")
    print("-" * 65)
    for i, g in enumerate(layer_groups):
        ls = ",".join(f"L{l}" for l in g["layers"])
        print(f"{i:<4} {ls:<38} {g['tier']:<8} {g['peak_fstat']:>7.2f} {g['sae_runs']:>5}")

    print("\nAdjacent-layer distances (combined, degrees):")
    for i in range(n_layers - 1):
        d = combined[i, i+1]
        if np.isnan(d):
            continue
        flag = " ← BOUNDARY" if d >= GROUP_THRESHOLD_DEG else ""
        attn_type = "GLOBAL" if i+1 in {4,9,14,19,24,29,34} else "local "
        print(f"  L{i:02d}↔L{i+1:02d}  [{attn_type}]  {d:5.1f}°{flag}")

    print("\nF-stat peaks by layer:")
    for l in range(n_layers):
        p    = peaks[l]
        tier = "HIGH  " if p >= TIER_HIGH else ("MED   " if p >= TIER_MEDIUM else "skip  ")
        bar  = "█" * min(40, int(p / 1.5))
        attn = "G" if l in {4,9,14,19,24,29,34} else "."
        print(f"  L{l:02d} [{attn}] {tier} F={p:7.2f}  {bar}")

    # ── 7. Save ───────────────────────────────────────────────────────────────
    out = {
        "threshold_deg": GROUP_THRESHOLD_DEG,
        "components": COMPONENTS,
        "groups": layer_groups,
        "total_sae_runs": total_runs,
        "fstat_peaks": {str(l): round(v, 4) for l, v in peaks.items()},
        "amad_combined": [
            [None if np.isnan(combined[i,j]) else round(float(combined[i,j]), 2)
             for j in range(n_layers)]
            for i in range(n_layers)
        ],
    }
    with open("layer_groups.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: layer_groups.json")


if __name__ == "__main__":
    main()
