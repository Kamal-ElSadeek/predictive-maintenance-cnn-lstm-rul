"""Recomputes the Fig. 10 predictions from the saved Stage 1 parallel model (stage1_b0005_parallel_model.h5).
Preprocessing is that of the Stage 1 notebook: Min-Max scaler fitted on cycles 1-49, window 10, one-step prediction.
Usage: python fig10_from_saved_model.py B0005.mat ../../04_battery_b0005/saved_models/stage1_b0005_parallel_model.h5 out.csv
Expected test RMSE: 0.047 Ah (as printed by the original run)."""
import sys, numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
mat = loadmat(sys.argv[1]); cyc = mat['B0005'][0, 0]['cycle'][0]
cap = np.array([float(c['data'][0][0]['Capacity'][0][0]) for c in cyc if c['type'][0] == 'discharge']); k = np.arange(1, len(cap) + 1)
tr, te = cap[k < 50], cap[k >= 50]
sc = MinMaxScaler((0, 1)).fit(tr.reshape(-1, 1))
inputs = sc.transform(np.r_[tr, te][len(tr) - 10:].reshape(-1, 1))
X = np.array([inputs[i - 10:i, 0] for i in range(10, 10 + len(te))]).reshape(-1, 10, 1)
p = sc.inverse_transform(tf.keras.models.load_model(sys.argv[2], compile=False).predict(X, verbose=0))[:, 0]
print('test RMSE %.4f Ah' % np.sqrt(np.mean((p - te) ** 2)))
np.savetxt(sys.argv[3], np.c_[np.arange(50, 50 + len(te)), p, te], delimiter=',', header='cycle,pred_Ah,actual_Ah', comments='', fmt='%.6f')
