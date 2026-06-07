# GWIQ-atlas (Geometry-Weighted Informed Quantization)

Model-agnostic brain-atlasing tools for transformer internals: build a per-layer map of what every channel in a language model does, store it in a queryable atlas, then use that map for surgical analysis and atlas-informed quantization.

GWIQ captures activation census signals, scores channels on a compliance/authentic behaviour axis, folds in layer-level SAE feature scores, and writes the result to a SQLite-backed atlas. Every feature/channel index maps 1:1 to a real weight-matrix row or column, so atlas scores can become quantization or intervention instructions.

Repo docs:

- [Contributing](./CONTRIBUTING.md)
- [Citation](./CITATION.cff)
- [Security](./SECURITY.md)
- [Code of Conduct](./CODE_OF_CONDUCT.md)
- [Requirements](./requirements.txt)

The supported direction is:

- No Modal requirement in core code.
- No hardcoded model identity in reusable modules.
- Model-specific runs live in config files or CLI arguments.
- Cloud runners and one-off probes stay out of the package core.

## Why the atlas exists

Standard PTQ methods decide which weights matter by calibrating on generic text. That works for perplexity and can quietly damage structure the calibration set never exercises, including compliance and persona behaviours. GWIQ measures what each channel does first, builds a durable map, then lets quantization read salience from the map instead of rediscovering it from a calibration corpus every time.

## Tool Map

### Core Atlas Pipeline

- **Activation census**: captures residual/MLP/attention projection activations by layer and component.
- **Layer analysis**: builds per-feature taxonomy, F-stat separation scores, coactivation pairs, heatmaps, and code/capability cross-reference.
- **Compliance behaviour map**: computes corporate-vs-authentic/compliance-axis F-stat, delta, mean_corp, and mean_auth per feature/channel.
- **Atlas builder**: merges census analysis, compliance behaviour maps, Sub-Zero reports, OV circuits, and SAE scores into a directory atlas plus SQLite mirror.
- **Query/index layer**: exposes cross-layer summaries and SQLite tables for surgical target selection.

### Sub-Zero / Surgery Tools

- **Aletheia**: gradient-guided sacred-layer ranking.
- **Forward capture**: collects residual and projection-input activations on corp/auth/neutral/red-team/capability corpora.
- **SVD decomposition**: decomposes projection matrices into candidate directions for per-projection intervention.
- **AtP gradient probe**: scores right-singular directions against the target behavior axis.
- **Cone / QR alignment**: replaces brittle max-cosine scoring with subspace projection norm against behavior cones.
- **Adaptive kneedle threshold**: finds a knee in sorted composite scores instead of using a fixed quantile.
- **Cross-layer coherence repass**: boosts directions that persist across neighboring layers/projections.
- **Causal ablation gate**: tests candidate directions with forward hooks and keeps only directions with signed positive behavioral effect.
- **DAS rotation**: rotates/intervenes in the discovered subspace for cleaner causal separation.
- **Capability fence**: rejects directions that damage protected capability/coding corpora.

### SAE Tools

- **SAE resolution pass**: maps residual-stream behavior signals into sparse dictionary features.
- **Surgical-gold query**: finds high-compliance behaviour / low-topic SAE features.
- **SAE eval suite**: reconstruction EV/L0/dead features, synthetic ground-truth recovery, cross-seed stability, and Soft-Frozen Decoder baseline.
- **AMAD screening**: behavioral-geometry angular distance over F-stat vectors for SAE budget allocation or layer grouping.

### Circuit / Quantization Tools

- **OV circuit analysis**: scores attention heads through W_OV structure, spectral concentration, effective rank, and compliance signal overlap.
- **Code/capability protection**: marks selective vs entangled capability channels so surgery and quantization can avoid collateral damage.
- **GWIQ / atlas-informed quantization**: uses atlas salience/deadness/capability masks as external importance signals for mixed precision or channel protection.
- **Mixed-precision channel protection**: public-facing quantization hook where atlas importance vectors can protect selected channels.

## Quantization Findings

These are the current Wikitext-2 results for Qwen3-8B-Base.

- **bf16 baseline**: `6.2192`
- **real bitsandbytes Q4**: `6.5421`
- **simulated NF4 anchor**: `6.5470`

What we tested:

- **Protect-only atlas mask**: `gwiq_protect_smart1230 = 6.3927`
- **Smart NF4 without protect**: `gwiq_nf4_smart1230 = 6.3999`
- **Best combined sweep point**: `gwiq_p130_m31 = 6.3897`
- **Other near-best points**: `gwiq_p130_m30 = 6.3900`, `gwiq_p125_m32 = 6.3905`, `gwiq_p135_m29 = 6.3914`

Failed paths:

- **Loose INT2 crush pools** exploded perplexity badly and were discarded.
- **Strict/calibrated INT2** was stable but still worse than real Q4.
- Conclusion: **INT2 is destructive on this model under the current atlas signals**.

Current conclusion:

- The best practical configuration is a mixed 4/8-bit scheme with **7,275 protect channels at INT8** plus atlas-aware smart NF4 for the rest.
- The current winner is `gwiq_p130_m31` at `6.3897`, which beats real Q4 and all prior mixed-precision baselines tested here.

## Current Core

- `qwip_atlas.config` defines `ModelSpec`, `CorpusSpec`, and `AtlasRunConfig`.
- `qwip_atlas.layers` resolves decoder layers and projection geometry across HF model families.
- `qwip_atlas.atlas_store` owns atlas layout, manifest helpers, and SQLite schema.
- `qwip_atlas.build_atlas` merges analysis products into the atlas directory and SQLite mirror.
- `qwip_atlas.analyze_layers` builds census taxonomy, F-stats, coactivation, heatmaps, and code/capability analysis.
- `qwip_atlas.analyze_compliance_behaviour` summarizes binary behavior-axis maps.
- `qwip_atlas.merge_sae` folds SAE score arrays into the atlas SQLite mirror.
- `qwip_atlas.extractors.local_census` captures multi-layer activation census locally with any compatible causal LM.
- `qwip_atlas.extractors.compliance_behaviour` computes local compliance-axis/compliance behaviour scores from two JSONL corpora.

## Install

```bash
pip install -e ".[analysis]"
```

## Build a census

```bash
qwip-atlas extract-local \
  --model <hf-model-or-local-path> \
  --corpus prompts.jsonl \
  --layers 30-35 \
  --outdir runs/example/census \
  --batch-size 8
```

Corpus is JSONL; each line needs at least a prompt and a category:

```json
{"prompt": "Explain recursion.", "category": "core_technical"}
```

Optional fields like `id`, `bucket`, `subcategory`, `is_contrast`, and `contrast_pair_id` are preserved.

## Compliance Behaviour Pass

```bash
qwip-atlas compliance-behaviour-local \
  --model <hf-model-or-local-path> \
  --positive corporate.jsonl \
  --negative authentic.jsonl \
  --layers 30-35 \
  --output runs/example/compliance_behaviour_scores.json
```

By convention, positive delta means the positive corpus is stronger. For the Sub-Zero/Bella use case, positive is usually corporate/compliance and negative is authentic/personality.

## Adding SAE Feature Scores

A complete atlas includes layer-level SAE feature scores alongside the census. GWIQ-atlas does not train the SAEs; it consumes scores trained elsewhere and folds them in with `qwip-merge-sae`.

If the model does not have SAEs yet, train them with [event-aware-SAE-trainer](https://github.com/JuiceB0xC0de/event-aware-SAE-trainer), then merge the resulting scores into the atlas. Without this step the atlas still works for census and compliance-behaviour queries; the SAE columns just stay empty.

## The Qwen3-8B-Base Atlas

The fully built atlas is intended to live on the Hub as `juiceb0xc0de/qwen3-8b-atlas` (`atlas.sqlite`, ~570MB). Check the repo for current availability and access.

A note on Qwen specifically: the compliance F-statistic is large nearly everywhere, so `fstat > 10` is not a useful filter the way it is on Gemma. Rank compliance by `|delta|` (mean_corp - mean_auth), not by fstat. Positive delta = compliance-leaning, negative = personality-leaning.

## Legacy

Modal-bound and model-specific scripts were removed from the package core. Reusable pieces should be ported into model-agnostic modules under `qwip_atlas/`.

## License

MIT - see [LICENSE](LICENSE).
