import os
import numpy as np

from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# ==================================================
# CONFIG
# ==================================================

MI_ROOT = "dataset"

SSVEP_ROOT = "ssvep_dataset_raw"

FS = 256

# ==================================================
# FEATURE EXTRACTION
# ==================================================

def extract_features(epoch):

    feats = []

    for ch in range(epoch.shape[0]):

        freqs, psd = welch(
            epoch[ch],
            fs=FS,
            nperseg=256
        )

        bands = [
            (4, 8),    # theta
            (8, 13),   # alpha
            (13, 30),  # beta
            (30, 45)   # gamma
        ]

        for low, high in bands:

            idx = (
                (freqs >= low)
                &
                (freqs < high)
            )

            feats.append(
                np.mean(psd[idx])
            )

    return np.array(feats)

# ==================================================
# LOAD MI
# ==================================================

X_all = []
y_all = []

for subject in ["P200", "P201"]:

    X = np.load(
        os.path.join(
            MI_ROOT,
            f"subject_{subject}",
            "X.npy"
        )
    )

    for epoch in X:

        X_all.append(
            extract_features(epoch)
        )

        y_all.append(0)

# ==================================================
# LOAD SSVEP
# ==================================================

for subject in ["P100", "P101"]:

    X = np.load(
        os.path.join(
            SSVEP_ROOT,
            f"subject_{subject}",
            "X.npy"
        )
    )

    X = X[:, :, :512]

    for epoch in X:

        X_all.append(
            extract_features(epoch)
        )

        y_all.append(1)

# ==================================================
# DATASET
# ==================================================

X_all = np.array(X_all)
y_all = np.array(y_all)

print("Dataset shape:")
print(X_all.shape)

print(
    "MI:",
    np.sum(y_all == 0)
)

print(
    "SSVEP:",
    np.sum(y_all == 1)
)

# ==================================================
# CV
# ==================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

preds = []
truth = []

for train_idx, test_idx in cv.split(X_all, y_all):

    X_train = X_all[train_idx]
    y_train = y_all[train_idx]

    X_test = X_all[test_idx]
    y_test = y_all[test_idx]

    clf = LinearDiscriminantAnalysis()

    clf.fit(
        X_train,
        y_train
    )

    y_pred = clf.predict(
        X_test
    )

    preds.extend(y_pred)
    truth.extend(y_test)

# ==================================================
# RESULTS
# ==================================================

preds = np.array(preds)
truth = np.array(truth)

acc = accuracy_score(
    truth,
    preds
)

cm = confusion_matrix(
    truth,
    preds
)

print("\nAccuracy:")
print(f"{acc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClass Accuracy:")

mi_acc = np.mean(
    preds[truth == 0] == 0
)

ssvep_acc = np.mean(
    preds[truth == 1] == 1
)

print(
    f"MI     : {mi_acc:.4f}"
)

print(
    f"SSVEP  : {ssvep_acc:.4f}"
)

print("\nDONE")