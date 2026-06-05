from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "sub-zero-atlas-probe"
DEFAULT_MODEL = "google/gemma-4-E2B-it"
DEFAULT_CORPORA_DIR = "/root/sub-zero/corpora"  # matches mount remote_path
DEFAULT_OUTPUT_DIR = "/data/outputs/sub-zero"
DEFAULT_OUT_NAME = "atlas-gemma4-e2b.pt"

import subprocess
from pathlib import Path

_repo = Path("/Users/chiggy/sub-zero")
if _repo.exists():
    COMMIT = subprocess.check_output(
        ["git", "-C", str(_repo), "rev-parse", "HEAD"]
    ).decode().strip()
else:
    COMMIT = "main"

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "gcc", "g++")
    .run_commands("python -m pip install --upgrade pip setuptools wheel")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "transformers>=4.51.0", "accelerate", "datasets", "sentencepiece",
        "protobuf", "hf_transfer", "scikit-learn", "scipy", "tqdm", "wandb", "jsonlines", "orjson", "jinja2",
    )
.run_commands(
    f"git clone https://github.com/juiceb0xc0de/sub-zero.git /root/sub-zero-repo && "
    f"cd /root/sub-zero-repo && git checkout {COMMIT} && "
    f"pip install -e /root/sub-zero-repo && "
    f"mkdir -p /root/sub-zero && "
    f"cp -r /root/sub-zero-repo/corpora /root/sub-zero/ 2>/dev/null || true && "
    f"ls /root/sub-zero/corpora/ 2>/dev/null || echo 'CORPORA NOT FOUND'",
    force_build=True:
    )
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

volume = modal.Volume.from_name("training_data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3 * 3600,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def build_sub_zero_atlas(
    model_name: str = DEFAULT_MODEL,
    corpora_dir: str = DEFAULT_CORPORA_DIR,
    out_dir: str = DEFAULT_OUTPUT_DIR,
    out_name: str = DEFAULT_OUT_NAME,
    max_prompts: int = 200,
    max_length: int = 256,
    layer_limit: int = 0,
    batch_size: int = 45,
    use_snmf: bool = True,
    snmf_components: int = 64,
    snmf_corp_auth_ratio: float = 2.0,
    snmf_cap_threshold: float = 0.3,
) -> dict:    
    
    import json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sub_zero.probe import ProbeConfig, build_atlas

    # ── CORPUS PATH FIX ──────────────────────────────────────────────────────
    # The pip install puts sub-zero at /root/sub-zero (via `pip install -e .`)
    # but the actual corpora live inside the cloned repo structure.
    # Try both locations to be resilient:
    possible_corpora_dirs = [
        corpora_dir,                                          # passed-in path
        "/root/sub-zero/corpora",                            # expected by probe
        "/root/sub-zero-repo/sub-zero/corpora",              # raw clone location
        "/root/sub-zero/sub-zero/corpora",                   # one level deeper
    ]
    chosen_corpora_dir = None
    for d in possible_corpora_dirs:
        p = Path(d)
        if p.exists() and any(p.iterdir()):
            chosen_corpora_dir = d
            break

    if chosen_corpora_dir is None:
        raise RuntimeError(
            f"Corpora not found in any of: {possible_corpora_dirs}\n"
            f"Contents of /root/sub-zero: {list(Path('/root/sub-zero').glob('*')) if Path('/root/sub-zero').exists() else 'N/A'}"
        )

    print(f"[sub-zero] using corpora from: {chosen_corpora_dir}")
    print(f"[sub-zero] corpora contents: {list(Path(chosen_corpora_dir).glob('*'))}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(out_dir) / out_name)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print(f"[sub-zero] loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token,
        dtype=dtype,
        low_cpu_mem_usage=False,
        device_map=None,
    )

    if torch.cuda.is_available():
        model = model.to("cuda")

    effective_max_prompts = None if max_prompts < 0 else max_prompts

    cfg = ProbeConfig(
          corpora_dir=chosen_corpora_dir,
          max_prompts_per_class=effective_max_prompts,
          max_length=max_length,
          layer_limit=layer_limit if layer_limit > 0 else None,
          batch_size=batch_size,
          use_snmf=use_snmf,
          snmf_components=snmf_components,
          snmf_corp_auth_ratio_threshold=snmf_corp_auth_ratio,
          snmf_capability_threshold=snmf_cap_threshold,
      )

    # Debug: print corpus stats, file locations, and cross-category overlap
    from sub_zero.probe import _read_lines, DEFAULT_CAPABILITY_CORPORA

    cap_corpora = cfg.capability_corpora or DEFAULT_CAPABILITY_CORPORA

    all_corpora = {
        "corporate":  (Path(chosen_corpora_dir) / cfg.corporate_file,  cfg.max_prompts_per_class),
        "authentic":  (Path(chosen_corpora_dir) / cfg.authentic_file,  cfg.max_prompts_per_class),
        "neutral":    (Path(chosen_corpora_dir) / cfg.neutral_file,    cfg.max_prompts_per_class),
        "red_team":   (Path(chosen_corpora_dir) / cfg.red_team_file,   cfg.max_prompts_per_class),
        **{name: (Path(chosen_corpora_dir) / fname, cfg.capability_max_prompts)
           for name, fname in cap_corpora.items()},
    }

    loaded: dict[str, list[str]] = {}
    print("[sub-zero] corpus inventory:")
    for name, (fpath, cap) in all_corpora.items():
        lines = _read_lines(fpath, cap)
        loaded[name] = lines
        status = "OK" if lines else "MISSING"
        required = " (required)" if name in ("corporate", "authentic") else ""
        print(f"  {name:<16} {len(lines):>3} prompts  {status}{required}")
        print(f"                   {fpath}")

    n_corp = len(loaded["corporate"])
    n_auth = len(loaded["authentic"])

    # Cross-category overlap check
    print("[sub-zero] overlap check:")
    names = list(loaded.keys())
    found_any = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = set(loaded[a]) & set(loaded[b])
            if overlap:
                found_any = True
                print(f"  WARNING {a} x {b}: {len(overlap)} shared prompt(s)")
                for p in list(overlap)[:3]:
                    print(f"    • {p[:80]!r}{'...' if len(p) > 80 else ''}")
    if not found_any:
        print("  no overlaps detected")

    if n_corp == 0 or n_auth == 0:
        raise RuntimeError(
            f"Need corporate + authentic corpora. "
            f"Found corp={n_corp}, auth={n_auth}. "
            f"Check files at {chosen_corpora_dir}"
        )

    import subprocess
    subprocess.run(["rm", "-f", out_path], check=False)

    # build task_batches for Aletheia from the corporate corpus
    corp_file = all_corpora["corporate"][0]
    corp_lines = loaded["corporate"]
    task_batches = []
    for i in range(0, len(corp_lines), cfg.batch_size):
        batch_texts = corp_lines[i:i + cfg.batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        enc["labels"] = enc["input_ids"].clone()
        task_batches.append(enc)
    print(f"[sub-zero] aletheia task_batches: {len(task_batches)} batches from {len(corp_lines)} corporate prompts")

    atlas = build_atlas(
        model=model,
        tokenizer=tokenizer,
        config=cfg,
        task_batches=task_batches,
        cache_path=out_path,
    )

    top_layers = []
    report_rows = []

    for li in sorted(atlas.layers.keys()):
        layer = atlas.layers[li]
        if not layer.per_projection:
            continue
        proj_rows = []
        layer_total = 0
        layer_bouncers = 0

        for pname, p in layer.per_projection.items():
            total = int(p.S.numel())
            bouncers = int(p.bouncer_sv_indices.numel())
            layer_total += total
            layer_bouncers += bouncers

            top_scores = []
            if bouncers > 0 and p.per_direction_classifier_score.numel() == p.S.numel():
                for idx in p.bouncer_sv_indices.tolist():
                    idx_i = int(idx)
                    top_scores.append(
                        {
                            "sv_index": idx_i,
                            "classifier_score": float(
                                p.per_direction_classifier_score[idx_i].item()
                            ),
                            "wanda_score": float(
                                p.per_direction_wanda_score[idx_i].item()
                            ),
                            "dark_variance": float(
                                p.per_direction_dark_variance[idx_i].item()
                            ),
                            "target_scale": float(
                                p.per_direction_target_scale[idx_i].item()
                            ),
                        }
                    )
                top_scores.sort(key=lambda r: r["classifier_score"], reverse=True)

            proj_rows.append(
                {
                    "projection": pname,
                    "sv_total": total,
                    "bouncer_sv": bouncers,
                    "bouncer_pct": (bouncers / total) if total > 0 else 0.0,
                    "top_bouncer_svs": top_scores,
                }
            )

        if layer_total == 0:
            continue

        layer_row = {
            "layer": int(li),
            "classifier_accuracy": float(layer.classifier_accuracy),
            "corp_refusal_angle_deg": float(layer.angle_degrees),
            "sv_total": layer_total,
            "bouncer_sv": layer_bouncers,
            "bouncer_pct": (layer_bouncers / layer_total),
            "projections": proj_rows,
        }
        report_rows.append(layer_row)
        top_layers.append((layer_row["bouncer_pct"], li, layer_bouncers, layer_total))

    top_layers.sort(reverse=True)
    top_layers = top_layers[:8]

    # ── Corpus decision map ──────────────────────────────────────────────────
    CORPUS_ROLES = {
        "corporate":  "bouncer detection  (compliance axis +)",
        "authentic":  "bouncer detection  (compliance axis −)",
        "neutral":    "baseline / normalisation",
        "red_team":   "refusal cone",
        **{name: "capability fence" for name in cap_corpora},
    }

    print()
    print("=" * 72)
    print("[sub-zero] CORPUS DECISION MAP")
    print("=" * 72)
    print()
    print("  CORPUS SOURCES")
    print(f"  {'name':<16} {'prompts':>7}  {'role':<38}  path")
    print(f"  {'-'*16}  {'-'*7}  {'-'*38}  {'-'*40}")
    for name, (fpath, _cap) in all_corpora.items():
        n = len(loaded[name])
        role = CORPUS_ROLES.get(name, "")
        print(f"  {name:<16} {n:>7}  {role:<38}  {fpath}")

    print()
    print("  LAYER BOUNCER HEATMAP")
    print("  (bouncer density per projection: █ ≥50%  ▓ ≥25%  ░ ≥10%  · <10%)")
    print()

    # Collect all projection names seen across all layers
    all_proj_names: list[str] = []
    for li in sorted(atlas.layers.keys()):
        for pname in atlas.layers[li].per_projection:
            if pname not in all_proj_names:
                all_proj_names.append(pname)

    header_projs = "  ".join(f"{p:<10}" for p in all_proj_names)
    print(f"  {'layer':<6}  {'acc':>5}  {'°':>5}  {header_projs}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*5}  " + "  ".join("-" * 10 for _ in all_proj_names))

    sacred_set = set(int(x) for x in atlas.sacred_layers)

    for li in sorted(atlas.layers.keys()):
        layer = atlas.layers[li]
        tag = " ◀ sacred" if li in sacred_set else ""
        acc_str = f"{layer.classifier_accuracy:.2f}" if layer.per_projection else "    -"
        ang_str = f"{layer.angle_degrees:.1f}" if layer.per_projection else "    -"

        proj_cells: list[str] = []
        for pname in all_proj_names:
            projat = layer.per_projection.get(pname)
            if projat is None or projat.S.numel() == 0:
                proj_cells.append(f"{'—':<10}")
                continue
            pct = projat.bouncer_sv_indices.numel() / projat.S.numel()
            if pct >= 0.50:
                sym = "█"
            elif pct >= 0.25:
                sym = "▓"
            elif pct >= 0.10:
                sym = "░"
            else:
                sym = "·"
            cell = f"{sym} {projat.bouncer_sv_indices.numel():>2}/{projat.S.numel():<5}"
            proj_cells.append(cell)

        row = "  ".join(proj_cells)
        print(f"  L{li:>02}    {acc_str:>5}  {ang_str:>5}  {row}{tag}")

    print()
    print("  TOP DECISION LAYERS  (by bouncer %)")
    for row in sorted(report_rows, key=lambda r: r["bouncer_pct"], reverse=True)[:8]:
        li = row["layer"]
        sacred_tag = "  [sacred]" if li in sacred_set else ""
        proj_summary = "  ".join(
            f"{p['projection']}={p['bouncer_sv']}/{p['sv_total']}"
            for p in row["projections"] if p["bouncer_sv"] > 0
        )
        print(
            f"  L{li:>02}  bouncer={row['bouncer_pct']:.3f} "
            f"({row['bouncer_sv']}/{row['sv_total']})  acc={row['classifier_accuracy']:.2f}"
            f"  [{proj_summary}]{sacred_tag}"
        )

    print()
    print("=" * 72)
    print()

    # ── JSON report ──────────────────────────────────────────────────────────
    report_path = str(Path(out_dir) / out_name.replace(".pt", "-report.json"))
    report = {
        "model": model_name,
        "atlas_path": out_path,
        "report_path": report_path,
        "hidden_size": int(atlas.hidden_size),
        "num_layers": int(atlas.num_layers),
        "sacred_layers": [int(x) for x in atlas.sacred_layers],
        "corpora_dir": chosen_corpora_dir,
        "corpus_stats": {
            name: {"prompts": len(lines), "path": str(fpath), "role": CORPUS_ROLES.get(name, "")}
            for name, (fpath, _cap), lines in (
                (n, all_corpora[n], loaded[n]) for n in all_corpora
            )
        },
        "top_layers_by_bouncer_pct": [
            {
                "layer": int(li),
                "bouncer_pct": float(pct),
                "bouncer_sv": int(bn),
                "sv_total": int(total),
            }
            for pct, li, bn, total in top_layers
        ],
        "layers": report_rows,
    }
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = {
        "model": model_name,
        "output": out_path,
        "report": report_path,
        "layers": atlas.num_layers,
        "sacred_layers": atlas.sacred_layers,
        "hidden_size": atlas.hidden_size,
        "layers_with_projection_data": len(report_rows),
        "corpora_dir": chosen_corpora_dir,
    }

    print(f"[sub-zero] detailed report saved: {report_path}")
    volume.commit()
    return summary


@app.local_entrypoint()
def main(
    model_name: str = DEFAULT_MODEL,
    corpora_dir: str = DEFAULT_CORPORA_DIR,
    out_dir: str = DEFAULT_OUTPUT_DIR,
    out_name: str = DEFAULT_OUT_NAME,
    max_prompts: int = 200,
    max_length: int = 256,
    layer_limit: int = 0,
    batch_size: int = 45,
    use_snmf: bool = True,
    snmf_components: int = 64,
    snmf_corp_auth_ratio: float = 2.0,
    snmf_cap_threshold: float = 0.3,
):
    result = build_sub_zero_atlas.remote(
        model_name=model_name,
        corpora_dir=corpora_dir,
        out_dir=out_dir,
        out_name=out_name,
        max_prompts=max_prompts,
        max_length=max_length,
        layer_limit=layer_limit,
        batch_size=batch_size,
        use_snmf=use_snmf,
        snmf_components=snmf_components,
        snmf_corp_auth_ratio=snmf_corp_auth_ratio,
        snmf_cap_threshold=snmf_cap_threshold,
    )

    print("\nRemote result:")
    print(result)
    print(f"\nAtlas path in Modal volume: {Path(out_dir) / out_name}")
    print("Run example: modal run a-modal-sub-zero-probe.py")
