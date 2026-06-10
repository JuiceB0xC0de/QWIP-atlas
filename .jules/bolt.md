## 2024-05-15 - [Vectorizing Array Reductions]
**Learning:** In the `qwip-atlas` codebase, where layer activations can be thousands of dimensions (e.g. `d_mlp` around 14336), computing boolean masks and taking array reductions (`mean`, `std`) inside a Python `for` loop over dimensions causes massive slow-downs (from 0.3s up to 10s per call).
**Action:** Always prioritize calculating aggregations and slice means over the entire tensor dimension across all rows prior to looping through individual rows, thereby doing operations once via NumPy's highly-optimized C backend.
## 2025-02-21 - [Vectorizing Nested Loops with Bitwise Operations]
**Learning:** In `compute_coactivation`, calculating a boolean mask of shape `[n_prompts]` inside nested loops `for r, c in pairs: for bkt in buckets:` caused severe performance drops via repeated re-allocations of large arrays.
**Action:** When computing sums over bitwise matches against categories, pre-compute a 2D boolean mask array `[n_buckets, n_prompts]` outside the loop. Use broadcasting and `.sum(axis=1)` to replace inner Python loops with a single C-level reduction.
