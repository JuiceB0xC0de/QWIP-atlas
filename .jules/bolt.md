## 2024-05-15 - [Vectorizing Array Reductions]
**Learning:** In the `qwip-atlas` codebase, where layer activations can be thousands of dimensions (e.g. `d_mlp` around 14336), computing boolean masks and taking array reductions (`mean`, `std`) inside a Python `for` loop over dimensions causes massive slow-downs (from 0.3s up to 10s per call).
**Action:** Always prioritize calculating aggregations and slice means over the entire tensor dimension across all rows prior to looping through individual rows, thereby doing operations once via NumPy's highly-optimized C backend.
## 2024-05-16 - [Delayed Tensor Operations]
**Learning:** In extraction scripts like `compliance_behaviour.py`, applying activation functions (`act_fn`) to an entire batch tensor before slicing out the desired token calculates values for `batch_size * seq_len` tokens instead of just the 1 needed token.
**Action:** Always slice PyTorch tensors to the exact subset needed (e.g., `tensor[batch_idx, sl][-1]`) *before* applying element-wise operations or reshapes.
