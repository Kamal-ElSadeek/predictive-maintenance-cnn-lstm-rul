# Architecture, Protocol and Search in Hybrid CNN–LSTM Prognostics

Code, run logs and figure scripts for the journal paper

> M. K. Habib and K. Mohamed, *Architecture, Protocol and Search in Hybrid CNN–LSTM Prognostics: A Controlled, Reproducible Evaluation on C-MAPSS, N-CMAPSS and NASA Battery Data* (under review).

The paper extends our ICMA conference paper. It follows a staged path. The stages and where their code lives:

| Stage (paper Fig. 3) | What it does | Folder |
|---|---|---|
| 1. Sequential vs parallel screen | Earlier protocol, one run per configuration; BO, GA and Hyperband on FD003, DS05 and B0005 | `01_stage1_screen/` |
| 2. Protocol correction | DS05 official split and representation (2×2×2 factorial); battery forecast protocols P1–P4 | `02_ncmapss_ds05_campaigns/` (v1–v7), `04_battery_b0005/` |
| 3. Search at equal budget | DS05 random / BO / PSO with two-stage selection; FD003 BO, PSO and L2 dose; B0005 60-trial BO | `02_…` (v8–v9), `03_cmapss_fd003_campaigns/`, `04_battery_b0005/` |
| 4. Constant screen and controls | 27-D fixed-constant screen, single-branch controls | `02_…` (v10–v15) |
| 5. Confirmatory re-test | All confirmatory results in the paper (Parts A–K) | `05_confirmatory_v16/`, `results/` |

## Layout

```
01_stage1_screen/            Stage 1 notebooks with their saved outputs (FD003, N-CMAPSS DS05, battery B0005)
02_ncmapss_ds05_campaigns/   N-CMAPSS DS05 campaigns v1–v15 (+ pooling ablation); logs/ holds the per-trial CSV/JSON
03_cmapss_fd003_campaigns/   C-MAPSS FD003 reconstruction of the fusion model of Peng et al. and campaigns v4–v9
04_battery_b0005/            Battery notebooks; logs/ holds protocol study, ablations, transfer and BO trial records
05_confirmatory_v16/         Review_Experiments_v16 notebook (Parts A–K), launcher, monitor and merge script
   pc2_package/              Same notebook configured for the second GPU (Parts F, G, I, J, K)
results/
   v16_labpc_rtx6000/        Parts A–E, G, H (Quadro RTX 6000): runs.jsonl, part CSVs, predictions
   v16_rtx3090/              Parts F, G, I, J, K (RTX 3090, TF32 off): runs.jsonl, part CSVs, executed notebook
   RESULTS_SUMMARY.md        Every number reported from the confirmatory runs
figures/
   scripts/                  make_figures.py, make_fig3_15.py, fig4c.py
   data/                     Inputs the scripts need besides the NASA data
   output/                   The result and schematic figures of the main text (SVG + 300 dpi PNG)
   stage1_original_outputs/  Figs. S1–S6 exactly as the Stage 1 notebooks produced them (kept unaltered for audit)
```

## Data

The NASA datasets are not redistributed here. See [DATA.md](DATA.md) for where to download them and where to put them.

## Environment

Confirmatory runs used Python 3.8, TensorFlow 2.10 (CUDA 11.2, cuDNN 8), NumPy 1.24, pandas 1.5, scikit-learn 1.3 and SciPy 1.10 on Windows; the exact versions of each run are in `results/*/env.json`. `environment.yml` recreates that environment. The Stage 1 notebooks (2023–2025) used TensorFlow 2.6 / Keras 2.6; see the header cell of each notebook.

```
conda env create -f environment.yml
conda activate cnnlstm-rul
```

## Reproducing the confirmatory results

1. Put the data in place (DATA.md).
2. Run `05_confirmatory_v16/Review_Experiments_v16.ipynb`, or on Windows `run_v16.bat`. The run store (`runs.jsonl`) is resumable: interrupted runs continue where they stopped. `V16_PARTS` selects parts (e.g. `A,C,H`) and `V16_RESULTS_DIR` sets the output folder.
3. Determinism: the notebook enables TensorFlow op determinism and disables TF32. Results agree across GPUs to five decimals on B0005 but can differ by up to 0.76 RMSE per seed on DS05, so the paper only compares runs made on the same GPU.
4. `merge_v16.py` merges the run stores of the two machines.

## Reproducing the figures

```
cd figures/scripts
python make_figures.py <DATA_DIR> ../output ../data/ncmapss_v9_all_trials.csv ../data/stage1_curves.json
python make_fig3_15.py ../output
python fig4c.py
```

`DATA_DIR` must contain `Nasa_battery_dataset/` (B0005.mat, `protocol_study_b0005.csv`, `bo_full_b0005/`); the last two are in `04_battery_b0005/logs/`. `stage1_curves.json` holds the loss histories printed in the Stage 1 notebooks; `make_figures.py` can replot them, but the paper uses the original notebook images in `stage1_original_outputs/` (Figs. S1, S2, S5 and S6 from the loss-plot cells of the Stage 1 notebooks, Figs. S3 and S4 as produced by the Stage 1 battery runs and carried unchanged from the earlier manuscript).

The schematic figures (Figs. 1, 3, 4 and 5) were drawn with AI assistance; every figure that shows data or results is plotted by the scripts here from the datasets and run logs.

## Notes on the Stage 1 material

The Stage 1 notebooks are kept unchanged, with their outputs, as the record of the screen. They use protocols that were later corrected (Section V-F of the paper), so their numbers are not comparable with the confirmatory results.

## Citation

See `CITATION.cff`. Please cite the journal paper once published.

## Licence

Code: MIT (see LICENSE). The NASA datasets are subject to the terms of the NASA Prognostics Data Repository.
