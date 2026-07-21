# C-MAPSS FD003 — Current-PC Checkpoint

## Inventory status

The currently accessible PC has been fully inventoried through Batch 06.

- Unique source notebooks retained: 8.
- Unique PDF exports retained: 3.
- Duplicate uploads were recorded but not stored again.
- All retained notebooks use NASA C-MAPSS FD003.

## Evidence located

- Sequential CNN–LSTM run: test RMSE 15.313, close to the thesis chapter's rounded 15.35.
- Parallel/GA-labelled submitted reference: test RMSE 15.032 and test R² 0.868, but no executed GA result.
- Parallel tuning runs: test RMSE values 17.493, 16.176, and 15.402.
- Earlier LSTM/GA development runs and failed GA attempts.

## Evidence still missing

- Executed parallel-model run reporting test RMSE 14.75.
- Executed GA-optimized run reporting test RMSE 14.98.
- Final C-MAPSS Hyperband output and configuration used in the thesis, if distinct.
- Original notebook cells or exported figures used for thesis Figures 43–45.
- Package/environment information for the final runs.
- Any saved final model weights, tuner directory, or concise run log that ties a configuration to the reported result.

## University-PC retrieval checklist

Search filenames and notebook contents for:

- `14.75`, `14.750`, `14.98`, and `14.980`.
- `FD003`, `Parallel`, `GA`, `Genetic`, `Hyperband`, `best_model`, and `best_deeper_model`.
- Notebook outputs containing `test set RMSE`, `Best Individual`, or `fitness (RMSE)`.
- Image files used for thesis Figures 43, 44, and 45.
- `requirements.txt`, Conda YAML files, tuner folders, model weights, and training logs.

Prefer the original `.ipynb` files with outputs intact. Data files may be supplied separately, but the public repository should ultimately document a lawful dataset-acquisition route rather than redistribute raw data without checking its terms.
