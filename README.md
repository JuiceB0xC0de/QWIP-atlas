# QWIP Atlas

Model-agnostic brain-atlasing tools for transformer internals — build a per-layer map of
what every channel in a language model *does*, and store it somewhere you can query it.

QWIP captures a per-layer activation census, scores every channel on a compliance↔authentic
behaviour axis, folds in layer-level SAE feature scores, and writes the whole thing to a
single queryable SQLite atlas. The atlas is the artifact: a map you can point at, where every
channel index maps 1:1 to a real weight-matrix row or column.

That map is built to be *used*. The first downstream experiment — run outside this package —
is **atlas-informed quantization that beats uniform Q4 on Qwen3-8B-Base, on the one metric
that's actively hostile to the idea.** Results are below.

## Why the atlas exists

Standard PTQ methods (AWQ, GPTQ) decide which weights matter by calibrating on a slab of
generic text. That works for perplexity and quietly destroys whatever the calibration set
never exercised — including behavioural structure like compliance and persona. QWIP takes the
opposite route: measure what each channel does *first*, build a durable map, then let
quantization read salience off the map instead of re-deriving it from a calibration corpus
every time.

The map for Qwen3-8B-Base covers all 36 layers, ~1.84M feature channels across 7 projections,
plus per-head attention stats, coactivation, and layer-level SAE feature scores (two sparsity
variants, trained separately and merged in — see [event-aware-SAE-trainer](https://github.com/JuiceB0xC0de/event-aware-SAE-trainer)
if you need to train SAEs for a model that doesn't have them yet). Every channel index maps
1:1 to a real weight-matrix row or column, so a salience score is also a quantization
instruction.

## GWIQ: geometry-weighted informed quantization

GWIQ is the quantization recipe built on the atlas. Per channel, it assigns a tier:

- **Protect** (INT8) — the most salient channels, kept near-lossless.
- **Standard** (NF4) — everything else.

A third **crush** tier (INT2 / ternary, targeting the lowest weight-L2-norm channels) was
tested and dropped — see results below. The current usable recipe is two-tier: protect + NF4.

The point isn't to win perplexity by a mile. It's that the atlas signal lets you place the
protect boundary *intelligently* rather than uniformly — and the channels GWIQ protects
include the behavioural ones that calibration-based methods would crush, because WikiText
never lit them up.

## Results: Qwen3-8B-Base, WikiText-2 perplexity

All numbers from `atlas11.py` on a Modal A100 (stride 512, max_len 2048). The fake-quant
simulator (real NF4 codebook, per-64-block absmax) reproduces real bitsandbytes Q4 to within
0.005 PPL, so every GWIQ number below is directly comparable to the bnb baseline.

| Config | PPL ↓ | Notes |
|--------|------:|-------|
| bf16 (reference) | 6.2192 | unquantized ceiling |
| **real bnb Q4 (NF4, double-quant)** | **6.5421** | **the bar to beat** |
| sim-NF4 uniform | 6.5470 | simulator anchor |
| **GWIQ `p130_m31`** | **6.3897** | **best observed — protect + smart NF4** |
| GWIQ `protect_smart1230` | 6.3927 | protect mask × smart NF4 |
| GWIQ `nf4_smart1230` | 6.3999 | smart NF4, no protect mask |
| GWIQ `atlas_protect` (protect-only) | 6.4301 | first atlas-informed win |

GWIQ's best run lands **0.15 PPL below real Q4** (6.3897 vs 6.5421) while keeping effective
bits in the same neighbourhood. The protect signal is real and consistently useful: every
configuration that kept the atlas protect mask on top of smart NF4 improved over the
mask-free version.

**The honest part:** perplexity is the *hard* test for GWIQ, not the flattering one. WikiText-2
exercises general language modelling, not compliance or persona — exactly the structure the
atlas is best at protecting. So a PPL win here is GWIQ winning on hostile ground; the payoff
the atlas is actually designed for (behavioural-eval fidelity after aggressive quantization)
is a separate measurement PPL can't show.

### What didn't work, and why it's informative

- **INT2 "crush" of census-dead channels: total collapse** (PPL ~29M, and the milder
  "loose" sweeps hit 60–178). The cause is a real trap worth stating plainly: a channel
  reading `activation_rate == 0` in the census is silent *on the atlas corpus*, not dead on
  general text. Early-layer gate projections are ~99% census-silent yet fire constantly on
  WikiText — ternarizing them cascades through every downstream layer. **Census silence does
  not license zeroing for a general-text metric.** Crush has to be driven by
  corpus-independent weight norm, not census activity.
- **Strict / calibration-gated crush: stable but still ≥ Q4** (6.59–6.66). Stopping the
  blow-up wasn't enough to win; INT2 crush under the signals tested here doesn't carry its
  weight.
- **The search plateaued** in the 6.3897–6.3959 band. Further gains need a new axis, not
  finer fraction sweeps around the winner.

The useful, reproducible recipe is therefore: **atlas protect mask (INT8) + smart NF4**, with
crush left out until a better crush signal exists.

## Repo layout

The canonical code lives under `qwip_atlas/`. Root-level scripts are being migrated into the
package; Modal-bound and model-specific scripts are archived under `legacy/`.

```
qwip_atlas/            # package-first core (model-agnostic, no Modal)
  config.py            #   ModelSpec, CorpusSpec, AtlasRunConfig
  layers.py            #   decoder-layer + projection geometry across HF families
  extractors/
    local_census.py    #   multi-layer activation census, any HF causal LM, local
build_atlas.py         # atlas SQLite management (migrating into package)
analyze_layers.py      # per-layer analysis (migrating into package)
analyze_bouncer.py     # compliance-behaviour scoring (migrating into package)
merge_sae.py           # fold externally-trained SAE feature scores into the atlas
amad_screening.py      # channel screening
legacy/                # Modal-bound + model-specific scripts, kept for reference
```

The migration direction: no Modal requirement in core code, no hardcoded model identity in
reusable modules, model-specific runs live in CLI args or config. Cloud runners and one-off
probes stay in `legacy/` until their useful pieces are ported.

## Install

```bash
pip install -e ".[hf,analysis]"
```

## Build a census

```bash
qwip-atlas extract-local \
  --model <hf-model-or-local-path> \
  --corpus prompts.jsonl \
  --layers 30-35 \
  --outdir runs/qwen3-8b/census \
  --batch-size 8
```

Corpus is JSONL; each line needs at least a prompt and a category:

```json
{"prompt": "Explain recursion.", "category": "core_technical"}
```

Optional fields (`id`, `bucket`, `subcategory`, `is_contrast`, `contrast_pair_id`) are
preserved through to the atlas.

## Adding SAE feature scores

A complete atlas includes layer-level SAE feature scores alongside the census. QWIP doesn't
train the SAEs — it consumes scores trained elsewhere and folds them in with `merge_sae.py`.
If the model you're atlasing doesn't have SAEs yet, train them with
[event-aware-SAE-trainer](https://github.com/JuiceB0xC0de/event-aware-SAE-trainer), which
trains an SAE on every decoder layer in a single unattended run, then merge the resulting
scores into the atlas. Without this step the atlas still works for census and
compliance-behaviour queries — the SAE columns just stay empty.

## The Qwen3-8B-Base atlas

The fully built atlas is intended to live on the Hub as `juiceb0xc0de/qwen3-8b-atlas`
(`atlas.sqlite`, ~570MB) — check the repo for current availability and access. `feature_idx`
is the real weight channel, so a query result is directly actionable on the model.

A note on Qwen specifically: the compliance F-statistic is large nearly everywhere
(85%+ of channels), so `fstat > 10` is *not* a useful filter the way it is on Gemma. Rank
compliance by `|delta|` (mean_corp − mean_auth), not by fstat. Positive delta =
compliance-leaning, negative = personality-leaning.

## License

MIT — see [LICENSE](LICENSE).
