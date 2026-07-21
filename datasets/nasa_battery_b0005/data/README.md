# NASA Battery B0005 data

The raw B0005 dataset is not redistributed in this repository.

The copy received during reconstruction had SHA-256 `2b14bfdd9ab935622cdd8562f39248ebf2a8d8f3701a17be0eefc2f8e41bd3f5`. It identifies as a MATLAB v5 file but its compressed stream ends unexpectedly, so it cannot be loaded reliably and is excluded.

Obtain a fresh authorized copy of `B0005.mat`, verify that it loads successfully, and keep its checksum in the rerun record. Use a strict chronological train/validation/test split and fit preprocessing only on the training cycles.
