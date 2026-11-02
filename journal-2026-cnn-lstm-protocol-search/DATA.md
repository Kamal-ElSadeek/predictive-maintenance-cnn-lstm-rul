# Data

All three datasets are public and come from the NASA Prognostics Center of Excellence Data Set Repository:
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

| Dataset | Files used | Expected location |
|---|---|---|
| C-MAPSS (turbofan, FD001–FD004) | `train_FD00x.txt`, `test_FD00x.txt`, `RUL_FD00x.txt` | `CMaps/` |
| N-CMAPSS (turbofan, DS02 and DS05) | `N-CMAPSS_DS02-006.h5`, `N-CMAPSS_DS05.h5` | `N-Cmapps/` |
| Li-ion battery aging (cells B0005, B0006, B0007, B0018) | `B0005.mat` … `B0018.mat` | `Nasa_battery_dataset/` |

The notebooks search for these folders under the working directory and its parents; the confirmatory notebook stops with a clear message if a file is missing (DS02 is optional and is skipped with a warning).
