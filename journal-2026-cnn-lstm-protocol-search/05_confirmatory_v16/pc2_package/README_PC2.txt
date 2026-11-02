SECOND PC (RTX 3090) - Parts F, G, I, J and K of the v16 review experiments
==========================================================================

1. Copy this whole PC2_package folder to the second PC.

2. Put the data in this folder or in any subfolder of it (the notebook finds it and prints where):
     N-CMAPSS_DS05.h5, N-CMAPSS_DS02-006.h5                 (Parts I, J, K)
     train_FD001..FD004.txt, test_FD001..FD004.txt,
     RUL_FD001..FD004.txt                                    (Parts J, K)
     B0005.mat, B0006.mat, B0007.mat, B0018.mat              (Part G)
     ncmapss_v12_all_trials.csv                              (Part F)

3. Double-click run_pc2.bat.
   - It uses the conda environment NCmapss_GPU. If the name differs, edit ENV_NAME at the top of run_pc2.bat.
   - It checks that TensorFlow sees the GPU, opens a monitor window, and runs about 17 hours.
   - If it stops, double-click it again: finished runs are kept and skipped.
   - When it finishes it writes results_v16_pc2_<date>.zip.

4. Merging (on the lab PC, after the second PC is finished):
   a. Stop the lab run (Ctrl+C in its window, answer Y).
   b. Unzip results_v16_pc2 next to the lab's results_v16 folder and copy merge_v16.py there.
   c. python merge_v16.py results_v16_pc2 results_v16
   d. Start run_v16.bat full on the lab PC again. It finishes Part E, skips F-K, and analyses everything (Part L).

Do not run the lab notebook and this one into the same results folder. The folder results_v16_pc2 already holds
the two selection files from the lab PC (Part B sequential configuration, Part H L2 value), so both PCs train
exactly the same configurations.
