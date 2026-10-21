import os
import numpy as np

from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# ==================================================
# CONFIG
# ==================================================

MI_ROOT = "dataset"
SSVEP_ROOT = "ssvep_dataset_raw"

FS = 256

# ==================================================
# FEATURES
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
            (4, 8),
            (8, 13),
            (13, 30),
            (30, 45)
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
# LOAD DATA
# ==================================================

X_all = []
y_all = []
groups_all = []

# ----------------------------
# MI
# ----------------------------

for subject in ["P200", "P201"]:

    X = np.load(
        os.path.join(
            MI_ROOT,
            f"subject_{subject}",
            "X.npy"
        )
    )

    groups = np.load(
        os.path.join(
            MI_ROOT,
            f"subject_{subject}",
            "groups.npy"
        )
    )

    for epoch, g in zip(X, groups):

        X_all.append(
            extract_features(epoch)
        )

        y_all.append(0)  # MI

        groups_all.append(
            f"MI_{subject}_{g}"
        )

# ----------------------------
# SSVEP
# ----------------------------

for subject in ["P100", "P101"]:

    X = np.load(
        os.path.join(
            SSVEP_ROOT,
            f"subject_{subject}",
            "X.npy"
        )
    )

    groups = np.load(
        os.path.join(
            SSVEP_ROOT,
            f"subject_{subject}",
            "groups.npy"
        )
    )

    X = X[:, :, :512]

    for epoch, g in zip(X, groups):

        X_all.append(
            extract_features(epoch)
        )

        y_all.append(1)  # SSVEP

        groups_all.append(
            f"SSVEP_{subject}_{g}"
        )

X_all = np.array(X_all)
y_all = np.array(y_all)
groups_all = np.array(groups_all)

# ==================================================
# SESSION-WISE CV
# ==================================================

unique_groups = np.unique(groups_all)

all_preds = []
all_truth = []

print("\nPer-session results\n")

for test_group in unique_groups:

    train_mask = groups_all != test_group
    test_mask = groups_all == test_group

    X_train = X_all[train_mask]
    y_train = y_all[train_mask]

    X_test = X_all[test_mask]
    y_test = y_all[test_mask]

    clf = LinearDiscriminantAnalysis()

    clf.fit(
        X_train,
        y_train
    )

    preds = clf.predict(
        X_test
    )

    acc = accuracy_score(
        y_test,
        preds
    )

    print(
        f"{test_group} : "
        f"{acc:.4f}"
    )

    all_preds.extend(preds)
    all_truth.extend(y_test)

# ==================================================
# RESULTS
# ==================================================

all_preds = np.array(all_preds)
all_truth = np.array(all_truth)

acc = accuracy_score(
    all_truth,
    all_preds
)

cm = confusion_matrix(
    all_truth,
    all_preds
)

mi_acc = np.mean(
    all_preds[all_truth == 0] == 0
)

ssvep_acc = np.mean(
    all_preds[all_truth == 1] == 1
)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print("\nAccuracy:")
print(f"{acc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClass Accuracy:")
print(f"MI     : {mi_acc:.4f}")
print(f"SSVEP  : {ssvep_acc:.4f}")

print("\nDONE")