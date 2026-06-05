"""
build_atlas.py
--------------
Merge tool for the master atlas. Subcommands:

  init           bootstrap an atlas dir with model_meta + (optional) corpus
  merge-layer    ingest one layer of census + analyzer outputs
  merge-subzero  fold a Sub-Zero atlas report into per-layer sub_zero/ subdirs
  index          rebuild the SQLite mirror + cross_layer/ summaries
  status         print a quick summary of what's in the atlas

Typical flow:

    python build_atlas.py --atlas atlas init \\
        --census l11_census_raw.json
    python build_atlas.py --atlas atlas merge-layer --layer 11 \\
        --census l11_census_raw.json --analysis-dir .
    python build_atlas.py --atlas atlas merge-subzero \\
        --report subzero-report.json
    python build_atlas.py --atlas atlas index
    python build_atlas.py --atlas atlas status
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

from atlas_lib import (
    DEFAULT_ATLAS_DIR, KNOWN_COMPONENTS, PER_HEAD_COMPONENTS,
    Manifest, atlas_paths, layer_paths,
    load_manifest, save_manifest, now_iso,
    open_sqlite, read_json, write_json, sha256_file,
    iter_layers, iter_components,
)


# ---------------------------------------------------------------------------
# init: bootstrap an atlas directory
# ---------------------------------------------------------------------------

def cmd_init(args):
    root = Path(args.atlas)
    paths = atlas_paths(root)
    for p in (paths["root"], paths["layers"], paths["cross_layer"]):
        p.mkdir(parents=True, exist_ok=True)

    # Read model meta from the census file's first record (if provided),
    # else use args-provided defaults.
    model_id   = args.model_id
    model_meta = {
        "d_mlp":      args.d_mlp,
        "d_model":    args.d_model,
        "n_heads":    args.n_heads,
        "n_kv_heads": args.n_kv_heads,
        "head_dim":   args.head_dim,
    }

    corpus_meta: dict = {}
    if args.census:
        census = read_json(args.census)
        n_prompts = len(census)
        buckets   = Counter(r.get("bucket", "?") for r in census)
        corpus_meta = {
            "source":    str(args.census),
            "hash":      sha256_file(Path(args.census)),
            "n_prompts": n_prompts,
            "buckets":   dict(buckets),
        }
        d_mlp = len(census[0].get("last_token", []))
        if d_mlp and not model_meta["d_mlp"]:
            model_meta["d_mlp"] = d_mlp

    manifest = Manifest(
        model_id=model_id,
        model_meta=model_meta,
        layers=[],
        components_per_layer={},
        subzero_layers=[],
        corpus=corpus_meta,
        updated_at=now_iso(),
    )
    save_manifest(root, manifest)
    write_json(paths["model_meta"], {"model_id": model_id, **model_meta})

    print(f"[init] atlas at {root}")
    print(f"[init] model: {model_id}  meta={model_meta}")
    if corpus_meta:
        print(f"[init] corpus: {corpus_meta['hash']}  n={corpus_meta['n_prompts']}  "
              f"buckets={len(corpus_meta['buckets'])}")


# ---------------------------------------------------------------------------
# merge-layer: ingest one layer's census + analyzer outputs
# ---------------------------------------------------------------------------

def _maybe_copy(src: Path, dst: Path):
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def cmd_merge_layer(args):
    root  = Path(args.atlas)
    layer = int(args.layer)
    L     = layer_paths(root, layer)
    analysis_dir = Path(args.analysis_dir)

    for p in (L["root"], L["components"], L["per_head"], L["sub_zero"]):
        p.mkdir(parents=True, exist_ok=True)

    # 1. Census raw copy + meta
    census_src = Path(args.census)
    if not census_src.exists():
        raise SystemExit(f"census file not found: {census_src}")
    no_copy = getattr(args, "no_census_copy", False)
    L["census_raw"].parent.mkdir(parents=True, exist_ok=True)
    if not no_copy and census_src.resolve() != L["census_raw"].resolve():
        shutil.copy2(census_src, L["census_raw"])      # skipped under --no-census-copy (avoids ~250GB of dupes)

    try:
        census = read_json(census_src)                 # orjson — fast even on a multi-GB census
        n_prompts = len(census)
    except Exception as e:
        print(f"[warn] census JSON unreadable ({e.__class__.__name__}): skipping corpus metadata")
        census = []
        n_prompts = None
    # cheap provenance marker under --no-census-copy; full content hash otherwise (sha256 of 7GB is slow)
    corpus_hash = f"size:{census_src.stat().st_size}" if no_copy else sha256_file(census_src)

    layer_meta = {
        "layer":         layer,
        "captured_at":   now_iso(),
        "corpus_hash":   corpus_hash,
        "n_prompts":     n_prompts,
        "components":    [],  # filled below
    }

    # 2. Per-component ingest. Files are named l<layer>_<comp>_<suffix>.
    components_present: list[str] = []
    for comp in KNOWN_COMPONENTS:
        prefix = f"l{layer}_{comp}_"
        tax_file = analysis_dir / f"{prefix}neuron_taxonomy.json"
        if not tax_file.exists():
            continue

        components_present.append(comp)
        cdir = L["components"] / comp
        cdir.mkdir(parents=True, exist_ok=True)

        # Files we expect from analyze_layers.py
        sources = {
            "taxonomy.json":      analysis_dir / f"{prefix}neuron_taxonomy.json",
            "separation.npy":     analysis_dir / f"{prefix}separation_scores.npy",
            "coactivation.json":  analysis_dir / f"{prefix}coactivation_pairs.json",
            "code_analysis.json": analysis_dir / f"{prefix}code_analysis.json",
        }
        copied = []
        for dst_name, src in sources.items():
            if _maybe_copy(src, cdir / dst_name):
                copied.append(dst_name)

        # Build a small summary.json for fast cross-cuts
        tax = read_json(cdir / "taxonomy.json")
        def _norm(c):
            return "specific" if c.startswith("specific_") else c
        tax_counts = Counter(_norm(c["class"]) for c in tax)

        sep = None
        if (cdir / "separation.npy").exists():
            sep = np.load(cdir / "separation.npy")

        code = read_json(cdir / "code_analysis.json") if (cdir / "code_analysis.json").exists() else {}

        summary = {
            "component":     comp,
            "n_features":    len(tax),
            "taxonomy":      dict(tax_counts),
            "fstat_top":     float(np.max(sep))  if sep is not None else None,
            "fstat_mean":    float(np.mean(sep)) if sep is not None else None,
            "fstat_top_idx": int(np.argmax(sep)) if sep is not None else None,
            "code_bucket":   code.get("code_bucket"),
            "code_entangled":code.get("entangled_count", 0),
            "code_selective":code.get("selective_count", 0),
            "files":         copied,
        }
        write_json(cdir / "summary.json", summary)

    # 3. Per-head ingest
    for ph in PER_HEAD_COMPONENTS:
        src = analysis_dir / f"l{layer}_{ph}_per_head.json"
        if src.exists():
            shutil.copy2(src, L["per_head"] / f"{ph}.json")

    # 4. Component-comparison passthrough (handy for cross_layer rebuilds)
    comp_cmp = analysis_dir / f"l{layer}_component_comparison.json"
    if comp_cmp.exists():
        shutil.copy2(comp_cmp, L["root"] / "component_comparison.json")

    layer_meta["components"] = components_present
    write_json(L["meta"], layer_meta)

    # 5. Update manifest
    m = load_manifest(root)
    if m is None:
        raise SystemExit("manifest missing — run `init` first")
    if layer not in m.layers:
        m.layers.append(layer)
        m.layers.sort()
    m.components_per_layer[layer] = components_present
    if not m.corpus and census:
        m.corpus = {
            "source":    str(census_src),
            "hash":      layer_meta["corpus_hash"],
            "n_prompts": layer_meta["n_prompts"],
            "buckets":   dict(Counter(r.get("bucket", "?") for r in census)),
        }
    m.updated_at = now_iso()
    save_manifest(root, m)

    print(f"[merge-layer] layer={layer}  components={components_present}")
    print(f"[merge-layer] corpus_hash={layer_meta['corpus_hash'][:20]}...  n={layer_meta['n_prompts']}")


# ---------------------------------------------------------------------------
# merge-subzero: fold Sub-Zero atlas report per-layer scores into atlas
# ---------------------------------------------------------------------------

def cmd_merge_subzero(args):
    root   = Path(args.atlas)
    report = read_json(Path(args.report))

    m = load_manifest(root)
    if m is None:
        raise SystemExit("manifest missing — run `init` first")

    sacred_layers = set(report.get("sacred_layers", []))
    subzero_layers = []

    for entry in report.get("layers", []):
        layer = int(entry["layer"])
        subzero_layers.append(layer)
        L = layer_paths(root, layer)
        L["sub_zero"].mkdir(parents=True, exist_ok=True)

        scores = {
            "layer":                  layer,
            "classifier_accuracy":    entry.get("classifier_accuracy"),
            "corp_refusal_angle_deg": entry.get("corp_refusal_angle_deg"),
            "sv_total":               entry.get("sv_total"),
            "bouncer_sv":             entry.get("bouncer_sv"),
            "bouncer_pct":            entry.get("bouncer_pct"),
            "is_sacred":              layer in sacred_layers,
        }
        write_json(L["subzero_scores"], scores)

        projs = entry.get("projections", [])
        write_json(L["subzero_projs"], projs)

        # If the layer has no meta yet (Sub-Zero touched a layer we haven't censused),
        # create a stub so the manifest accounts for it.
        if not L["meta"].exists():
            L["root"].mkdir(parents=True, exist_ok=True)
            write_json(L["meta"], {
                "layer":       layer,
                "captured_at": now_iso(),
                "components":  [],  # no census yet
                "stub":        True,
            })

    # Stitch into manifest
    for layer in subzero_layers:
        if layer not in m.layers:
            m.layers.append(layer)
    m.layers.sort()
    m.subzero_layers = sorted(subzero_layers)

    # Also push model_meta from report (hidden_size + num_layers)
    if report.get("hidden_size") and not m.model_meta.get("d_model"):
        m.model_meta["d_model"] = report["hidden_size"]
    if report.get("num_layers") and not m.model_meta.get("num_layers"):
        m.model_meta["num_layers"] = report["num_layers"]

    m.updated_at = now_iso()
    save_manifest(root, m)
    print(f"[merge-subzero] folded {len(subzero_layers)} layers from {args.report}")
    print(f"[merge-subzero] sacred layers: {sorted(sacred_layers)}")


# ---------------------------------------------------------------------------
# merge-bouncer: fold bouncer_scores.json (corporate-vs-authentic) into atlas
# ---------------------------------------------------------------------------

def cmd_merge_bouncer(args):
    root   = Path(args.atlas)
    report = read_json(Path(args.report))

    m = load_manifest(root)
    if m is None:
        raise SystemExit("manifest missing — run `init` first")

    n_layers_in = 0
    for L_str, comps in report.items():
        layer = int(L_str)
        L = layer_paths(root, layer)
        bdir = L["root"] / "bouncer"
        bdir.mkdir(parents=True, exist_ok=True)

        per_layer_summary = {
            "layer":         layer,
            "n_corporate":   None,
            "n_authentic":   None,
            "components":    {},
        }

        for comp, data in comps.items():
            cdir = bdir / comp
            cdir.mkdir(parents=True, exist_ok=True)

            fstat = np.asarray(data["fstat"],    dtype=np.float32)
            delta = np.asarray(data["delta"],    dtype=np.float32)
            mean_c = np.asarray(data["mean_corp"], dtype=np.float32)
            mean_a = np.asarray(data["mean_auth"], dtype=np.float32)
            std_c  = np.asarray(data["std_corp"],  dtype=np.float32)
            std_a  = np.asarray(data["std_auth"],  dtype=np.float32)

            np.save(cdir / "fstat.npy",     fstat)
            np.save(cdir / "delta.npy",     delta)
            np.save(cdir / "mean_corp.npy", mean_c)
            np.save(cdir / "mean_auth.npy", mean_a)
            np.save(cdir / "std_corp.npy",  std_c)
            np.save(cdir / "std_auth.npy",  std_a)

            # Top-K snapshot for quick reading
            top_idx = np.argsort(fstat)[-30:][::-1]
            top = [{
                "feature":   int(i),
                "fstat":     float(fstat[i]),
                "delta":     float(delta[i]),
                "mean_corp": float(mean_c[i]),
                "mean_auth": float(mean_a[i]),
                "leaning":   "corp" if delta[i] > 0 else "auth",
            } for i in top_idx]

            head_dim    = data.get("head_dim")
            is_per_head = bool(data.get("is_per_head"))
            summary = {
                "component":    comp,
                "n_features":   int(fstat.shape[0]),
                "fstat_top":    float(fstat.max()),
                "fstat_mean":   float(fstat.mean()),
                "fstat_top_idx":int(fstat.argmax()),
                "delta_at_top": float(delta[int(fstat.argmax())]),
                "is_per_head":  is_per_head,
                "head_dim":     head_dim,
                "top_features": top,
            }
            write_json(cdir / "summary.json", summary)

            per_layer_summary["n_corporate"] = int(data["n_corporate"])
            per_layer_summary["n_authentic"] = int(data["n_authentic"])
            per_layer_summary["components"][comp] = {
                "fstat_top":    summary["fstat_top"],
                "fstat_mean":   summary["fstat_mean"],
                "delta_at_top": summary["delta_at_top"],
                "top_feature":  summary["fstat_top_idx"],
                "is_per_head":  is_per_head,
                "head_dim":     head_dim,
            }

        write_json(bdir / "summary.json", per_layer_summary)
        n_layers_in += 1

    m.updated_at = now_iso()
    save_manifest(root, m)
    print(f"[merge-bouncer] folded {n_layers_in} layers of bouncer scores from {args.report}")


# ---------------------------------------------------------------------------
# merge-ov: fold ov_circuit_scores.json into atlas/layers/<N>/ov/
# ---------------------------------------------------------------------------

def cmd_merge_ov(args):
    root    = Path(args.atlas)
    records = read_json(Path(args.report))

    m = load_manifest(root)
    if m is None:
        raise SystemExit("manifest missing — run `init` first")

    by_layer: dict[int, list] = {}
    for r in records:
        by_layer.setdefault(r["layer"], []).append(r)

    for layer, heads in by_layer.items():
        ov_dir = layer_paths(root, layer)["root"] / "ov"
        ov_dir.mkdir(parents=True, exist_ok=True)
        write_json(ov_dir / "heads.json", heads)

    m.updated_at = now_iso()
    save_manifest(root, m)
    print(f"[merge-ov] folded {len(records)} head records across {len(by_layer)} layers from {args.report}")


# ---------------------------------------------------------------------------
# index: rebuild SQLite mirror + cross_layer/ summaries
# ---------------------------------------------------------------------------

def _exec(conn: sqlite3.Connection, sql: str, params=()):
    conn.execute(sql, params)


def cmd_index(args):
    root = Path(args.atlas)
    m = load_manifest(root)
    if m is None:
        raise SystemExit("manifest missing — run `init` first")

    # Fresh SQLite (we're indexing, not appending)
    db_path = atlas_paths(root)["sqlite"]
    if db_path.exists():
        db_path.unlink()
    conn = open_sqlite(root)

    sacred_set = set(m.subzero_layers) & set()  # placeholder; sacred_set populated below
    # Reload sacred set from any subzero/scores.json
    sacred_set = set()
    for layer in iter_layers(root):
        sp = layer_paths(root, layer)["subzero_scores"]
        if sp.exists() and read_json(sp).get("is_sacred"):
            sacred_set.add(layer)

    layers_rows = []
    features_rows = []
    per_head_rows = []
    subzero_layer_rows = []
    subzero_sv_rows = []
    coact_rows = []
    code_rows  = []
    bouncer_feat_rows = []
    bouncer_head_rows = []
    ov_rows = []

    for layer in iter_layers(root):
        L = layer_paths(root, layer)
        meta = read_json(L["meta"]) if L["meta"].exists() else {}

        has_census  = bool(meta.get("components"))
        sz_scores   = read_json(L["subzero_scores"]) if L["subzero_scores"].exists() else None
        has_subzero = sz_scores is not None

        layers_rows.append((
            layer, m.model_id, meta.get("n_prompts"),
            meta.get("corpus_hash"), meta.get("captured_at"),
            int(has_census), int(has_subzero), int(layer in sacred_set),
        ))

        if has_subzero:
            subzero_layer_rows.append((
                layer,
                sz_scores.get("classifier_accuracy"),
                sz_scores.get("corp_refusal_angle_deg"),
                sz_scores.get("sv_total"),
                sz_scores.get("bouncer_sv"),
                sz_scores.get("bouncer_pct"),
            ))
            projs = read_json(L["subzero_projs"]) if L["subzero_projs"].exists() else []
            for p in projs:
                proj_name = p.get("projection")
                for sv in p.get("top_bouncer_svs", []):
                    subzero_sv_rows.append((
                        layer, proj_name, sv.get("sv_index"),
                        sv.get("classifier_score"), sv.get("wanda_score"),
                        sv.get("dark_variance"), sv.get("target_scale"),
                    ))

        # Per-component features + coactivation + code
        for comp in iter_components(root, layer):
            cdir = L["components"] / comp
            tax = read_json(cdir / "taxonomy.json")
            sep = np.load(cdir / "separation.npy") if (cdir / "separation.npy").exists() else None

            for rec in tax:
                idx = int(rec["neuron_idx"])
                features_rows.append((
                    layer, comp, idx,
                    rec.get("class"),
                    rec.get("activation_rate"),
                    rec.get("mean_activation"),
                    rec.get("std_activation"),
                    float(sep[idx]) if sep is not None and idx < len(sep) else None,
                ))

            coact_file = cdir / "coactivation.json"
            if coact_file.exists():
                for pair in read_json(coact_file):
                    coact_rows.append((
                        layer, comp, pair.get("neuron_a"), pair.get("neuron_b"),
                        pair.get("correlation"), pair.get("dominant_bucket"),
                    ))

            code_file = cdir / "code_analysis.json"
            if code_file.exists():
                code = read_json(code_file)
                for idx in code.get("entangled_neurons", []):
                    code_rows.append((layer, comp, int(idx), "entangled"))
                for idx in code.get("selective_neurons", []):
                    code_rows.append((layer, comp, int(idx), "selective"))

        # Per-head
        if L["per_head"].exists():
            for ph_file in L["per_head"].glob("*.json"):
                comp = ph_file.stem
                heads = read_json(ph_file)
                for h in heads:
                    tax = h.get("taxonomy", {})
                    per_head_rows.append((
                        layer, comp, h.get("head"),
                        h.get("dims"), tax.get("specific", 0),
                        h.get("top_sep_score"), h.get("mean_sep_score"),
                        h.get("top_code_dim"), h.get("top_code_spec"),
                    ))

        # OV circuit scores
        ov_file = L["root"] / "ov" / "heads.json"
        if ov_file.exists():
            for h in read_json(ov_file):
                ov_rows.append((
                    layer, h["head"], h.get("kv_head"),
                    h.get("top_singular_val"), h.get("total_energy"),
                    h.get("spectral_conc"), h.get("eff_rank"),
                    h.get("compliance_score"), h.get("layer_comp_strength"),
                    json.dumps(h.get("top3_sv", [])),
                ))

        # Bouncer features + per-head from atlas/layers/<N>/bouncer/<comp>/
        bdir = L["root"] / "bouncer"
        if bdir.exists():
            for sub in bdir.iterdir():
                if not sub.is_dir():
                    continue
                comp = sub.name
                fstat = np.load(sub / "fstat.npy") if (sub / "fstat.npy").exists() else None
                delta = np.load(sub / "delta.npy") if (sub / "delta.npy").exists() else None
                if fstat is None or delta is None:
                    continue
                mean_c = np.load(sub / "mean_corp.npy")
                mean_a = np.load(sub / "mean_auth.npy")
                # Insert all features (cheap — sub-50K rows per layer/comp)
                for idx in range(len(fstat)):
                    bouncer_feat_rows.append((
                        layer, comp, int(idx),
                        float(fstat[idx]), float(delta[idx]),
                        float(mean_c[idx]), float(mean_a[idx]),
                    ))
                # If per-head: derive head-level rows
                summ = read_json(sub / "summary.json")
                if summ.get("is_per_head") and summ.get("head_dim"):
                    head_dim = int(summ["head_dim"])
                    if len(fstat) % head_dim == 0:
                        H = len(fstat) // head_dim
                        fs2 = fstat.reshape(H, head_dim)
                        d2  = delta.reshape(H, head_dim)
                        for h in range(H):
                            top = int(fs2[h].argmax())
                            bouncer_head_rows.append((
                                layer, comp, h, head_dim,
                                float(fs2[h, top]), float(fs2[h].mean()),
                                float(d2[h, top]), top,
                                int(d2[h, top] > 0),
                            ))

    # Bulk insert
    cur = conn.cursor()
    cur.executemany("INSERT OR REPLACE INTO layers VALUES (?,?,?,?,?,?,?,?)", layers_rows)
    cur.executemany("INSERT OR REPLACE INTO features VALUES (?,?,?,?,?,?,?,?)", features_rows)
    cur.executemany("INSERT OR REPLACE INTO per_head VALUES (?,?,?,?,?,?,?,?,?)", per_head_rows)
    cur.executemany("INSERT OR REPLACE INTO subzero_layer VALUES (?,?,?,?,?,?)", subzero_layer_rows)
    cur.executemany("INSERT OR REPLACE INTO subzero_svs   VALUES (?,?,?,?,?,?,?)", subzero_sv_rows)
    cur.executemany("INSERT INTO coactivation VALUES (?,?,?,?,?,?)", coact_rows)
    cur.executemany("INSERT OR REPLACE INTO code_analysis VALUES (?,?,?,?)", code_rows)
    cur.executemany("INSERT OR REPLACE INTO bouncer_features VALUES (?,?,?,?,?,?,?)", bouncer_feat_rows)
    cur.executemany("INSERT OR REPLACE INTO bouncer_per_head VALUES (?,?,?,?,?,?,?,?,?)", bouncer_head_rows)
    cur.executemany("INSERT OR REPLACE INTO ov_circuits VALUES (?,?,?,?,?,?,?,?,?,?)", ov_rows)
    conn.commit()
    conn.close()

    # cross_layer/ summaries (small JSON snapshots, easy diff in git)
    cross = atlas_paths(root)["cross_layer"]
    cross.mkdir(parents=True, exist_ok=True)

    tax_by_layer: dict[int, dict] = {}
    fstat_by_layer: dict[int, dict] = {}
    code_by_layer: dict[int, dict] = {}

    for layer in iter_layers(root):
        L = layer_paths(root, layer)
        tax_by_layer[layer]   = {}
        fstat_by_layer[layer] = {}
        code_by_layer[layer]  = {}
        for comp in iter_components(root, layer):
            summ = read_json(L["components"] / comp / "summary.json")
            tax_by_layer[layer][comp]   = summ.get("taxonomy")
            fstat_by_layer[layer][comp] = {
                "top":     summ.get("fstat_top"),
                "mean":    summ.get("fstat_mean"),
                "top_idx": summ.get("fstat_top_idx"),
            }
            code_by_layer[layer][comp] = {
                "code_bucket":    summ.get("code_bucket"),
                "code_entangled": summ.get("code_entangled"),
                "code_selective": summ.get("code_selective"),
            }

    write_json(cross / "taxonomy_by_layer.json",         tax_by_layer)
    write_json(cross / "fstat_top_by_layer.json",        fstat_by_layer)
    write_json(cross / "code_entanglement_by_layer.json", code_by_layer)

    print(f"[index] {len(layers_rows)} layers, {len(features_rows)} features, "
          f"{len(per_head_rows)} per-head rows, {len(subzero_sv_rows)} SVs, "
          f"{len(bouncer_feat_rows)} bouncer features, {len(bouncer_head_rows)} bouncer heads, "
          f"{len(ov_rows)} OV circuit rows")
    print(f"[index] sqlite: {db_path}")


# ---------------------------------------------------------------------------
# status: quick overview
# ---------------------------------------------------------------------------

def cmd_status(args):
    root = Path(args.atlas)
    m = load_manifest(root)
    if m is None:
        print(f"[status] no manifest at {root}")
        return

    print(f"[status] atlas: {root}")
    print(f"[status] model: {m.model_id}  meta={m.model_meta}")
    if m.corpus:
        print(f"[status] corpus: {m.corpus.get('hash', '')[:30]}  "
              f"n={m.corpus.get('n_prompts')}  buckets={len(m.corpus.get('buckets', {}))}")
    print(f"[status] layers: {m.layers}")
    print(f"[status] sub_zero layers: {m.subzero_layers}")
    print(f"[status] components_per_layer:")
    for layer, comps in sorted(m.components_per_layer.items()):
        print(f"           layer {layer}: {comps}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas", default=str(DEFAULT_ATLAS_DIR))
    sub = p.add_subparsers(dest="cmd", required=True)

    # init
    s = sub.add_parser("init")
    s.add_argument("--model-id", default="unknown")
    s.add_argument("--d-mlp",      type=int, default=None)
    s.add_argument("--d-model",    type=int, default=None)
    s.add_argument("--n-heads",    type=int, default=None)
    s.add_argument("--n-kv-heads", type=int, default=None)
    s.add_argument("--head-dim",   type=int, default=None)
    s.add_argument("--census", default=None,
                   help="optional path to a census JSON to register corpus hash + buckets")
    s.set_defaults(func=cmd_init)

    # merge-layer
    s = sub.add_parser("merge-layer")
    s.add_argument("--layer", type=int, required=True)
    s.add_argument("--census", required=True,
                   help="path to l<N>_census_raw.json")
    s.add_argument("--analysis-dir", default=".",
                   help="dir containing analyzer outputs (l<N>_*_neuron_taxonomy.json etc.)")
    s.add_argument("--no-census-copy", action="store_true",
                   help="don't duplicate the multi-GB census into the atlas, and use a cheap "
                        "size marker instead of a full sha256 (keeps the atlas lean + merge fast)")
    s.set_defaults(func=cmd_merge_layer)

    # merge-subzero
    s = sub.add_parser("merge-subzero")
    s.add_argument("--report", required=True,
                   help="path to Sub-Zero atlas report JSON")
    s.set_defaults(func=cmd_merge_subzero)

    # merge-bouncer
    s = sub.add_parser("merge-bouncer")
    s.add_argument("--report", required=True,
                   help="path to bouncer_scores.json from extract_bouncer.py")
    s.set_defaults(func=cmd_merge_bouncer)

    # merge-ov
    s = sub.add_parser("merge-ov")
    s.add_argument("--report", required=True,
                   help="path to ov_circuit_scores.json")
    s.set_defaults(func=cmd_merge_ov)

    # index
    s = sub.add_parser("index")
    s.set_defaults(func=cmd_index)

    # status
    s = sub.add_parser("status")
    s.set_defaults(func=cmd_status)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
