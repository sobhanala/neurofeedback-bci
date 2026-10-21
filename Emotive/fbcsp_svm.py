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
    "P100",
    "P101"
]

SFREQ = 256

BANDS = [
    (8,12),
    (12,16),
    (16,20),
    (20,24),
    (24,30)
]

CSP_COMPONENTS = 2

N_SELECTED_FEATURES = 6

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

print("\n" + "="*60)
print("TRUE FBCSP + LINEAR SVM")
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

    feature_blocks = []

    for band in BANDS:

        low, high = band

        print(f"Band {low}-{high} Hz")

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

    print(
        "Raw feature shape:",
        X_features.shape
    )

    selector = SelectKBest(
        mutual_info_classif,
        k=min(
            N_SELECTED_FEATURES,
            X_features.shape[1]
        )
    )

    X_selected = selector.fit_transform(
        X_features,
        y
    )

    print(
        "Selected feature shape:",
        X_selected.shape
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

    print(
        f"Accuracy: "
        f"{np.mean(scores):.3f} ± "
        f"{np.std(scores):.3f}"
    )

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)