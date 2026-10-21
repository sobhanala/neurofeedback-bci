# train_mi_ssvep_gate.py

import os
import numpy as np
import pickle
import json

from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# =========================
# CONFIG
# =========================

MI_ROOT = "dataset"
SSVEP_ROOT = "ssvep_dataset_raw"

FS = 256

MODEL_DIR = "gate_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# FEATURES
# =========================

def extract_psd_features(epoch):

    # epoch: (14, 512)
    feats = []

    for ch in range(epoch.shape[0]):

        freqs, psd = welch(epoch[ch], fs=FS, nperseg=256)

        bands = [
            (4, 8),
            (8, 13),
            (13, 30),
            (30, 45)
        ]

        for low, high in bands:
            idx = (freqs >= low) & (freqs < high)
            feats.append(np.mean(psd[idx]))

    return np.array(feats)

# =========================
# LOAD DATA
# =========================

X = []
y = []

# MI = 0
for sub in ["P200", "P201"]:

    data = np.load(f"{MI_ROOT}/subject_{sub}/X.npy")

    for epoch in data:
        X.append(extract_psd_features(epoch))
        y.append(0)

# SSVEP = 1
for sub in ["P100", "P101"]:

    data = np.load(f"{SSVEP_ROOT}/subject_{sub}/X.npy")

    data = data[:, :, :512]

    for epoch in data:
        X.append(extract_psd_features(epoch))
        y.append(1)

X = np.array(X)
y = np.array(y)

print("Dataset:", X.shape)
print("MI:", np.sum(y == 0))
print("SSVEP:", np.sum(y == 1))

# =========================
# TRAIN MODEL
# =========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = LinearDiscriminantAnalysis()
clf.fit(X_scaled, y)

# =========================
# SAVE
# =========================

with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open(f"{MODEL_DIR}/lda.pkl", "wb") as f:
    pickle.dump(clf, f)

metadata = {
    "fs": FS,
    "bands": [
        (4, 8),
        (8, 13),
        (13, 30),
        (30, 45)
    ],
    "input_shape": "14x512",
    "classes": {
        "0": "MI",
        "1": "SSVEP"
    }
}

with open(f"{MODEL_DIR}/model_info.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("\nMODEL SAVED → gate_model/")