import numpy as np

from scipy.signal import butter, filtfilt

from mne.decoding import CSP

from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# =====================================
# CONFIG
# =====================================

DATASET_ROOT = "dataset"

SUBJECTS = [
    "P200",
    "P201"
]

SFREQ = 256

BANDS = [
    (8,12),
    (12,16),
    (16,20),
    (20,24),
    (24,30)
]

#CSP_COMPONENTS = 2
#CSP_COMPONENTS = 4
#CSP_COMPONENTS = 6
CSP_COMPONENTS = 10
#FEATURE_COUNTS = [
#    2,
#    4,
#    6,
#    8,
#    10
#]
#FEATURE_COUNTS = [4,8,12,16]
#FEATURE_COUNTS = [6,12,18,24]
#FEATURE_COUNTS = [
#    24,
#    28,
#    30
#]
FEATURE_COUNTS = [
    20,
    30,
    40,
    50
]
# =====================================
# BANDPASS FILTER
# =====================================

def bandpass_epoch(epoch, low, high, fs):

    nyq = fs / 2

    b, a = butter(
        4,
        [low/nyq, high/nyq],
        btype='band'
    )

    return filtfilt(
        b,
        a,
        epoch,
        axis=1
    )

# =====================================
# MAIN
# =====================================

FINAL_RESULTS = {}

print("\n" + "="*60)
print("FBCSP FEATURE SWEEP")
print("="*60)

for sub in SUBJECTS:

    print(f"\n===== SUBJECT {sub} =====")

    X = np.load(
        f"{DATASET_ROOT}/subject_{sub}/X.npy"
    )

    y = np.load(
        f"{DATASET_ROOT}/subject_{sub}/y.npy"
    )

    groups = np.load(
        f"{DATASET_ROOT}/subject_{sub}/groups.npy"
    )

    # ---------------------------------
    # Build FBCSP Features
    # ---------------------------------

    feature_blocks = []

    for low, high in BANDS:

        X_band = np.array([
            bandpass_epoch(
                epoch,
                low,
                high,
                SFREQ
            )
            for epoch in X
        ])

        csp = CSP(
            n_components=CSP_COMPONENTS,
            reg='ledoit_wolf',
            log=True
        )

        X_csp = csp.fit_transform(
            X_band,
            y
        )

        feature_blocks.append(
            X_csp
        )

    X_features = np.concatenate(
        feature_blocks,
        axis=1
    )

    print("Raw feature shape:", X_features.shape)

    best_acc = 0
    best_std = 0
    best_k = None

    FINAL_RESULTS[sub] = []

    for k in FEATURE_COUNTS:

        selector = SelectKBest(
            mutual_info_classif,
            k=min(k, X_features.shape[1])
        )

        X_selected = selector.fit_transform(
            X_features,
            y
        )

        svm = SVC(
            kernel='linear'
        )

        logo = LeaveOneGroupOut()

        scores = cross_val_score(
            svm,
            X_selected,
            y,
            cv=logo.split(
                X_selected,
                y,
                groups
            )
        )

        acc = np.mean(scores)
        std = np.std(scores)

        print(
            f"k={k:2d} | "
            f"Accuracy={acc:.3f} ± {std:.3f}"
        )

        FINAL_RESULTS[sub].append(
            (k, acc, std)
        )

        if acc > best_acc:
            best_acc = acc
            best_std = std
            best_k = k

    print(
        f"BEST FOR {sub}: "
        f"k={best_k} | "
        f"Accuracy={best_acc:.3f} ± {best_std:.3f}"
    )

# =====================================
# FINAL SUMMARY
# =====================================

print("\n")
print("="*70)
print("FINAL SUMMARY")
print("="*70)

for sub in SUBJECTS:

    print(f"\n{sub}")

    for k, acc, std in FINAL_RESULTS[sub]:

        print(
            f"k={k:2d} | "
            f"{acc:.3f} ± {std:.3f}"
        )

    best = max(
        FINAL_RESULTS[sub],
        key=lambda x: x[1]
    )

    print(
        f"BEST => "
        f"k={best[0]} | "
        f"{best[1]:.3f} ± {best[2]:.3f}"
    )

print("\n" + "="*70)
print("SWEEP COMPLETE")
print("="*70)