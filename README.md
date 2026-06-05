# QWIP Atlas

Model-agnostic brain atlasing tools for transformer internals.

This repo is being cleaned up from working research scripts. The supported direction is:

- No Modal requirement in core code.
- No hardcoded model identity in reusable modules.
- Model-specific runs live in config files or CLI arguments.
- Cloud runners, old one-off probes, and historical scripts live under `legacy/`.

## Current Core

- `qwip_atlas.config` defines `ModelSpec`, `CorpusSpec`, and `AtlasRunConfig`.
- `qwip_atlas.layers` resolves decoder layers and projection geometry across HF model families.
- `qwip_atlas.extractors.local_census` captures multi-layer activation census locally with any compatible causal LM.
- `build_atlas.py`, `analyze_layers.py`, and `atlas_lib.py` are still being migrated into the package.

## Example

```bash
pip install -e ".[hf,analysis]"

qwip-atlas extract-local \
  --model <hf-model-or-local-path> \
  --corpus prompts.jsonl \
  --layers 30-35 \
  --outdir runs/qwen3-8b/census \
  --batch-size 8
```

The corpus JSONL should contain at least:

```json
{"prompt": "Explain recursion.", "category": "core_technical"}
```

Optional fields like `id`, `bucket`, `subcategory`, `is_contrast`, and `contrast_pair_id` are preserved.

## Legacy

Modal-bound and model-specific scripts are archived in `legacy/modal/` until their useful pieces are ported into model-agnostic modules.
