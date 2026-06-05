# Legacy Scripts

These scripts are preserved for provenance while QWIP Atlas is converted into a
model-agnostic, local/package-first repo.

`legacy/modal/` contains old Modal apps and model-specific runners. Treat them as
reference implementations only. Supported code should move shared logic into
`qwip_atlas/` and expose model identity through config or CLI arguments.
