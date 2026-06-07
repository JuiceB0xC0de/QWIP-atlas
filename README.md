```markdown
# GWIQ-atlas (Geometry-Weighted Informed Quantization)

Model-agnostic brain-atlasing tools for transformer internals — build a per-layer map of what every channel in a language model does, store
it in a queryable atlas, and use that map for surgical analysis and quantization.

Standard PTQ methods decide which weights matter by calibrating on generic text. That works for perplexity, but it quietly destroys whatever
the calibration set never exercised — including compliance and persona behaviors.

GWIQ takes the opposite route: measure what each channel does first, build a durable map, and let quantization read salience off the map
instead of re-deriving it from a calibration corpus every time.

Every channel index in the atlas maps 1:1 to a real weight-matrix row or column, so a salience score is also a quantization instruction.

Repo docs: [Contributing](./CONTRIBUTING.md) · [Citation](./CITATION.cff) · [Security](./SECURITY.md) · [Code of
Conduct](./CODE_OF_CONDUCT.md) · [Requirements](./requirements.txt)

**Design principles:**
- No Modal requirement in core code.
- No hardcoded model identity in reusable modules.
- Model-specific runs live in config files or CLI arguments.
- Cloud runners and one-off probes stay out of the package core.

---

## The Results

GWIQ's best run lands **0.15 PPL below real Q4** (6.3897 vs 6.5421) on Qwen3-8B-Base while keeping effective bits in the same neighborhood.
The protect signal is real and consistently useful: every configuration that kept the atlas protect mask on top of smart NF4 improved over
the mask-free version.

| Config | PPL ↓ | Notes |
|--------|-------|-------|
| bf16 (reference) | 6.2192 | Unquantized ceiling |
| real bnb Q4 (NF4, double-quant) | 6.5421 | The bar to beat |
| simulated NF4 uniform | 6.5470 | Simulator anchor |
| **gwiq_p130_m31** | **6.3897** | **Best observed — protect + smart NF4** |
| gwiq_protect_smart1230 | 6.3927 | Protect mask × smart NF4 |
| gwiq_nf4_smart1230 | 6.3999 | Smart NF4, no protect mask |
| gwiq_atlas_protect (protect-only) | 6.4301 | First atlas-informed win |

**The honest part:** Perplexity is the hard test for GWIQ, not the flattering one. WikiText-2 exercises general language modeling, not
compliance or persona — exactly the structure the atlas is best at protecting. A PPL win here is GWIQ winning on hostile ground. The payoff
the atlas is actually designed for (behavioral-eval fidelity after aggressive quantization) is a separate measurement PPL can't show.

### What didn't work

- **INT2 "crush" of census-dead channels:** Total collapse (PPL ~29M, milder sweeps hit 60–178). A channel reading `activation_rate == 0` in
the census is silent on the atlas corpus, not dead on general text. Early-layer gate projections are ~99% census-silent yet fire constantly
on WikiText. Census silence does not license zeroing for a general-text metric. Crush must be driven by corpus-independent weight norm, not
census activity.
- **Strict/calibration-gated crush:** Stable but still ≥ Q4 (6.59–6.66). INT2 crush under the signals tested here doesn't carry its weight.

The search plateaued in the 6.3897–6.3959 band. Further gains need a new axis, not finer fraction sweeps around the winner.

**The practical recipe:** Atlas protect mask (INT8) + smart NF4, with crush left out until a better crush signal exists.

---

## The Pipeline

The atlas isn't a single script; it's a pipeline of tools that build on each other.

### 1. Core Atlas Pipeline (Build the Map)
- **Activation census:** Captures residual/MLP/attention projection activations by layer and component.
- **Layer analysis:** Builds per-feature taxonomy, F-stat separation scores, coactivation pairs, heatmaps, and code/capability
cross-reference.
- **Compliance behavior map:** Computes corporate-vs-authentic/compliance-axis F-stat, delta, mean_corp, and mean_auth per feature/channel.
- **Atlas builder:** Merges census analysis, compliance maps, Sub-Zero reports, OV circuits, and SAE scores into a directory atlas plus
SQLite mirror.
- **Query/index layer:** Exposes cross-layer summaries and SQLite tables for surgical target selection.

### 2. Sub-Zero / Surgery Tools (Find the Targets)
- **Aletheia:** Gradient-guided sacred-layer ranking.
- **Forward capture:** Collects residual and projection-input activations on corp/auth/neutral/red-team/capability corpora.
- **SVD decomposition:** Decomposes projection matrices into candidate directions for per-projection intervention.
- **AtP gradient probe:** Scores right-singular directions against the target behavior axis.
- **Cone / QR alignment:** Replaces brittle max-cosine scoring with subspace projection norm against behavior cones.
- **Adaptive kneedle threshold:** Finds a knee in sorted composite scores instead of using a fixed quantile.
- **Cross-layer coherence repass:** Boosts directions that persist across neighboring layers/projections.
- **Causal ablation gate:** Tests candidate directions with forward hooks and keeps only directions with signed positive behavioral effect.
- **DAS rotation:** Rotates/intervenes in the discovered subspace for cleaner causal separation.
- **Capability fence:** Rejects directions that damage protected capability/coding corpora.

### 3. SAE Tools (The Bass Player)
Every solo guitarist needs a rhythm section. The SAE tools map residual-stream behavior signals into sparse dictionary features so the atlas
can see fine-grained structure.
- **SAE resolution pass:** Maps residual-stream behavior signals into sparse dictionary features.
- **Surgical-gold query:** Finds high-compliance behavior / low-topic SAE features.
- **SAE eval suite:** Reconstruction EV/L0/dead features, synthetic ground-truth recovery, cross-seed stability, and Soft-Frozen Decoder
baseline.
- **AMAD screening:** Behavioral-geometry angular distance over F-stat vectors for SAE budget allocation or layer grouping.

*(Train SAEs with [event-aware-SAE-trainer](https://github.com/JuiceB0xC0de/event-aware-SAE-trainer), then merge scores into the atlas.)*

### 4. Circuit / Quantization Tools (Do the Surgery)
- **OV circuit analysis:** Scores attention heads through W_OV structure, spectral concentration, effective rank, and compliance signal
overlap.
- **Code/capability protection:** Marks selective vs entangled capability channels so surgery and quantization avoid collateral damage.
- **GWIQ / atlas-informed quantization:** Uses atlas salience/deadness/capability masks as external importance signals for mixed precision
or channel protection.
- **Mixed-precision channel protection:** Public-facing quantization hook where atlas importance vectors can protect selected channels.

---

## Quick Start

### Install
```bash
pip install -e ".[analysis]"
```

### Build a Census
```bash
qwip-atlas extract-local \
  --model <hf-model-or-local-path> \
  --corpus prompts.jsonl \
  --layers 30-35 \
  --outdir runs/example/census \
  --batch-size 8
```

Corpus is JSONL; each line needs at least a `prompt` and `category`:
```json
{"prompt": "Explain recursion.", "category": "core_technical"}
```
Optional fields like `id`, `bucket`, `subcategory`, `is_contrast`, and `contrast_pair_id` are preserved.

### Compliance Behavior Pass
```bash
qwip-atlas compliance-behaviour-local \
  --model <hf-model-or-local-path> \
  --positive corporate.jsonl \
  --negative authentic.jsonl \
  --layers 30-35 \
  --output runs/example/compliance_behaviour_scores.json
```

By convention, positive delta means the positive corpus is stronger. For the Sub-Zero/Bella use case, positive is usually
corporate/compliance and negative is authentic/personality.

### Adding SAE Feature Scores
A complete atlas includes layer-level SAE feature scores alongside the census. GWIQ-atlas does not train the SAEs; it consumes scores
trained elsewhere and folds them in with `qwip-merge-sae`.

If the model does not have SAEs yet, train them with [event-aware-SAE-trainer](https://github.com/JuiceB0xC0de/event-aware-SAE-trainer),
then merge the resulting scores into the atlas. Without this step the atlas still works for census and compliance-behaviour queries; the SAE
columns just stay empty.

---

## The Qwen3-8B-Base Atlas

The fully built atlas is intended to live on the Hub as `juiceb0xc0de/qwen3-8b-atlas` (`atlas.sqlite`, ~570MB). Check the repo for current
availability and access.

**A note on Qwen specifically:** The compliance F-statistic is large nearly everywhere, so `fstat > 10` is not a useful filter the way it is
on Gemma. Rank compliance by `|delta|` (`mean_corp - mean_auth`), not by fstat. Positive delta = compliance-leaning, negative =
personality-leaning.

---

## Current Core

- `qwip_atlas.config` defines `ModelSpec`, `CorpusSpec`, and `AtlasRunConfig`.
- `qwip_atlas.layers` resolves decoder layers and projection geometry across HF model families.
- `qwip_atlas.atlas_store` owns atlas layout, manifest helpers, and SQLite schema.
- `qwip_atlas.build_atlas` merges analysis products into the atlas directory and SQLite mirror.
- `qwip_atlas.analyze_layers` builds census taxonomy, F-stats, coactivation, heatmaps, and code/capability analysis.
- `qwip_atlas.analyze_compliance_behaviour` summarizes binary behavior-axis maps.
- `qwip_atlas.merge_sae` folds SAE score arrays into the atlas SQLite mirror.
- `qwip_atlas.extractors.local_census` captures multi-layer activation census locally with any compatible causal LM.
- `qwip_atlas.extractors.compliance_behaviour` computes local compliance-axis/compliance behavior scores from two JSONL corpora.

## License

MIT — see [LICENSE](LICENSE).
```