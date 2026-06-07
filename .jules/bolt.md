## 2024-05-18 - Avoid redundant full-batch activations in local_census
**Learning:** `_row_from_capture` in `local_census.py` computed the activation function on the FULL batched `gate_pre` tensor for every single sequence in the batch. This recalculates `SiLU` (an expensive op) O(batch_size^2) times effectively, and processes padding tokens unnecessarily.
**Action:** Always slice tensors to the specific sequence (and remove padding) BEFORE applying element-wise operations like activation functions, especially inside loops over batch indices.
