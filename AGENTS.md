# AGENTS.md — GWIQ-atlas / qwip-atlas

## What this repo is

GWIQ-atlas is a model-agnostic brain-atlasing toolkit for transformer internals. It builds a
per-layer, per-channel map of what every projection does in a language model — activation
census, compliance↔authentic behaviour scoring, SAE feature scores, attention stats — and
stores it in a queryable SQLite atlas. That map drives GWIQ (Geometry-Weighted Informed
Quantization), which has beaten real bitsandbytes Q4 on Qwen3-8B-Base by 0.15 PPL on
WikiText-2 — a metric deliberately hostile to atlas-informed methods.

The canonical package is `qwip_atlas/`. Root-level scripts (`build_atlas.py`,
`analyze_layers.py`, `analyze_bouncer.py`, `merge_sae.py`, `amad_screening.py`) are being
migrated into the package. `legacy/` holds Modal-bound and model-specific scripts archived
for reference.

## Project conventions

**Python ≥ 3.11.** All new code lives under `qwip_atlas/`. No Modal dependency in core code.
No hardcoded model identity in reusable modules — model identity belongs in CLI args or
config objects (`ModelSpec`, `AtlasRunConfig`). Cloud runners stay in `legacy/` until their
useful pieces are ported.

Install:
```bash
pip install -e ".[hf,analysis]"
```

## Architecture

```
qwip_atlas/
  config.py          # ModelSpec, CorpusSpec, AtlasRunConfig (frozen dataclasses)
  layers.py          # decoder-layer + projection geometry across HF model families
  cli.py             # qwip-atlas CLI (extract-local + compliance-behaviour-local subcommands)
  io.py              # atlas I/O helpers
  extractors/
    local_census.py           # multi-layer activation census, any HF causal LM, local GPU
    compliance_behaviour.py   # binary compliance-behaviour extraction pipeline for ModelSpec/AtlasRunConfig
```

The atlas is SQLite. Every channel index maps 1:1 to a real weight-matrix row or column, so
a salience score is directly a quantization instruction. `feature_idx` is the real weight
channel throughout.

SAE scores come from outside this package — train them with
[event-aware-SAE-trainer](https://github.com/JuiceB0xC0de/event-aware-SAE-trainer) and fold
them in with `merge_sae.py`.

## Where things live, and what's still migrating

| File | Status |
|------|--------|
| `qwip_atlas/` | Canonical — edit here |
| `build_atlas.py` | Migrating into `qwip_atlas/` |
| `analyze_layers.py` | Migrating into `qwip_atlas/` |
| `analyze_bouncer.py` | Migrating into `qwip_atlas/` |
| `merge_sae.py` | Migrating into `qwip_atlas/` |
| `amad_screening.py` | Migrating into `qwip_atlas/` |
| `legacy/` | Read-only reference; do not add new code here |

When porting root-level scripts, strip out Modal dependencies and hardcoded model IDs.
Expose functionality through `cli.py` subcommands or importable functions.

## Key domain facts for agents

- **Census silence ≠ dead.** A channel with `activation_rate == 0` on the atlas corpus still
  fires on general text (esp. early-layer gate projections). Do not treat census-zero channels
  as candidates for zeroing or INT2 crush — the INT2 experiment collapsed to ~29M PPL for
  exactly this reason.
- **Qwen-specific:** compliance F-statistic is large nearly everywhere (85%+ of channels).
  Filter by `|delta|` (mean_corp − mean_auth), not `fstat`. Positive delta = compliance-lean,
  negative = personality-lean.
- **GWIQ recipe:** protect (INT8) + smart NF4. The three-tier protect+NF4+INT2-crush config
  was tested and abandoned — do not reintroduce it without a corpus-independent crush signal
  (weight-L2-norm, not census activity).
- **Perplexity is the hostile metric.** WikiText-2 exercises general language modelling, not
  compliance or persona. A PPL win here is GWIQ winning on unfavourable ground.

## Corpus format

JSONL, one prompt per line. Required fields:

```json
{"prompt": "...", "category": "core_technical"}
```

Optional: `id`, `bucket`, `subcategory`, `is_contrast`, `contrast_pair_id`.

## Running a census

```bash
qwip-atlas extract-local \
  --model <hf-model-or-local-path> \
  --corpus prompts.jsonl \
  --layers 0-35 \
  --outdir runs/mymodel/census \
  --batch-size 8
```

`HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` env vars are picked up automatically, or pass
`--hf-token`.

## Testing

No formal test suite yet. When adding features, manually verify:
1. `qwip-atlas extract-local` runs end-to-end on a small model (e.g. `Qwen/Qwen3-0.6B`) with
   a minimal corpus (5–10 prompts, 2–3 layers).
2. Output `.npz` files have expected keys and shapes.
3. No hardcoded model identity or Modal imports in `qwip_atlas/` code.

## Related repos

- [event-aware-SAE-trainer](https://github.com/JuiceB0xC0de/event-aware-SAE-trainer) — trains
  SAEs on every decoder layer in one unattended run; output feeds `merge_sae.py`.
- Hub dataset: `juiceb0xc0de/qwen3-8b-atlas` — the fully built Qwen3-8B-Base atlas (~570MB SQLite).