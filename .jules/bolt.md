## 2024-05-15 - [Vectorizing Array Reductions]
**Learning:** In the `qwip-atlas` codebase, where layer activations can be thousands of dimensions (e.g. `d_mlp` around 14336), computing boolean masks and taking array reductions (`mean`, `std`) inside a Python `for` loop over dimensions causes massive slow-downs (from 0.3s up to 10s per call).
**Action:** Always prioritize calculating aggregations and slice means over the entire tensor dimension across all rows prior to looping through individual rows, thereby doing operations once via NumPy's highly-optimized C backend.

## 2024-06-25 - [Tensor Optimization: Slice before operation]
**Learning:** In PyTorch extraction pipelines, computing activation functions (`act_fn`) or reshaping logic across full tensors (e.g. `O(batch_size * seq_len)`) before subsetting them to extract one sequence token causes enormous redundant compute.
**Action:** When extracting specific tensor data, slice the tensor down first (`tensor[batch_idx, sl][-1]`) and *then* apply mathematical operations or reshapes to the vastly smaller resultant slice.
