# Contributing to QWIP Atlas

QWIP Atlas is a research-to-product cleanup of a larger brain-atlas stack. The
goal is to keep the core package model-agnostic, reproducible, and easy to run
locally or in Modal without the old one-off scripts leaking back into the main
path.

## What belongs here

- Package-native code under `qwip_atlas/`.
- Thin compatibility shims only when they are needed to preserve an existing
  command path.
- README, docs, tests, and metadata that make the repo easier to use.
- Narrow, well-scoped experiments that improve the atlas or quantization path.

## What does not belong here

- New hardcoded model-specific behavior in shared modules.
- Modal-only logic in package modules unless it is clearly isolated.
- Duplicate wrappers when the package entrypoint already exists.
- Large generated artifacts, model checkpoints, or raw atlas outputs.

## Development flow

1. Make the smallest useful change.
2. Keep public APIs stable where possible.
3. Prefer package entrypoints and CLI args over hardcoded paths.
4. Run the smallest relevant verification step before opening a larger sweep.
5. Update docs when a workflow or command changes.

## Suggested checks

- `python -m py_compile qwip_atlas/*.py`
- `python -m qwip_atlas.cli --help`
- The relevant benchmark or extraction command for the area you touched.

## Style

- Use ASCII unless the file already uses Unicode.
- Keep comments short and technical.
- Prefer explicit names over abbreviations in shared code.
- Preserve compatibility wrappers only when they reduce user friction.

## Questions worth answering in a PR

- Does this make the atlas easier to rebuild or query?
- Does this reduce model specificity?
- Does this improve reproducibility, not just performance?
- Is the change documented well enough for the next person to run it without
  re-reading the source?
