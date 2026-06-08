## 2024-05-18 - Avoid redundant full-batch activations in local_census
**Learning:** `_row_from_capture` in `local_census.py` computed the activation function on the FULL batched `gate_pre` tensor for every single sequence in the batch. This recalculates `SiLU` (an expensive op) O(batch_size^2) times effectively, and processes padding tokens unnecessarily.
**Action:** Always slice tensors to the specific sequence (and remove padding) BEFORE applying element-wise operations like activation functions, especially inside loops over batch indices.
## 2023-10-27 - [ANOVA Identity in Python Numpy]
**Learning:** For-loops inside numpy arrays for within group variance are slow and use excessive memory.
**Action:** Use SST = SSB + SSW to avoid for loops calculating within group variance. Calculate SST over the entire array, calculate SSB across groups, then subtract to get SSW.
