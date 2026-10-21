# evaluate_ssvep_harmonic_ratio.py

import os
import numpy as np

from scipy.signal import welch
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# ==================================================
# CONFIG
# ==================================================

DATASET_ROOT = "ssvep_dataset_raw"

SUBJECTS = [
    "P100",
    "P101"
]

FS = 256

# O1 / O2 positions
O1_IDX = 6
O2_IDX = 7

# ==================================================
# HELPERS
# ==================================================

def get_power(freqs, psd, target_freq):

    idx = np.argmin(
        np.abs(freqs - target_freq)
    )

    return psd[idx]


def harmonic_score(freqs, psd, base_freq):

    harmonics = [
        base_freq,
        base_freq * 2,
        base_freq * 3
    ]

    score = 0.0

    for h in harmonics:

        score += get_power(
            freqs,
            psd,
            h
        )

    return score


# ==================================================
# MAIN
# ==================================================

all_results = []

for subject in SUBJECTS:

    print("\n" + "=" * 60)
    print(f"SUBJECT {subject}")
    print("=" * 60)

    subject_path = os.path.join(
        DATASET_ROOT,
        f"subject_{subject}"
    )

    X = np.load(
        os.path.join(
            subject_path,
            "X.npy"
        )
    )

    y = np.load(
        os.path.join(
            subject_path,
            "y_freq.npy"
        )
    )
    groups = np.load(
        os.path.join(
            subject_path,
            "groups.npy"
        )
    )
    
    if subject == "P100":
    
        mask = groups != "S004"
    
        X = X[mask]
        y = y[mask]
    
        print(
            f"Removed S004. Remaining trials: {len(y)}"
        )    

    predictions = []

    class12_score12 = []
    class12_score15 = []

    class15_score12 = []
    class15_score15 = []

    for epoch, label in zip(X, y):

        # O1 + O2

        eeg = epoch[
            [O1_IDX, O2_IDX],
            :
        ]

        eeg = eeg.mean(
            axis=0
        )

        freqs, psd = welch(
            eeg,
            fs=FS,
            nperseg=1024
        )

        score12 = harmonic_score(
            freqs,
            psd,
            12
        )

        score15 = harmonic_score(
            freqs,
            psd,
            15
        )

        pred = (
            0
            if score12 > score15
            else 1
        )

        predictions.append(pred)

        if label == 0:

            class12_score12.append(
                score12
            )

            class12_score15.append(
                score15
            )

        else:

            class15_score12.append(
                score12
            )

            class15_score15.append(
                score15
            )

    predictions = np.array(
        predictions
    )

    acc = accuracy_score(
        y,
        predictions
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    print("\nAccuracy:")
    print(f"{acc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nMean Harmonic Scores")

    print(
        f"12Hz trials -> "
        f"S12={np.mean(class12_score12):.6f} "
        f"S15={np.mean(class12_score15):.6f}"
    )

    print(
        f"15Hz trials -> "
        f"S12={np.mean(class15_score12):.6f} "
        f"S15={np.mean(class15_score15):.6f}"
    )

    all_results.append({
        "subject": subject,
        "accuracy": acc,
        "s12_12": np.mean(class12_score12),
        "s15_12": np.mean(class12_score15),
        "s12_15": np.mean(class15_score12),
        "s15_15": np.mean(class15_score15)
    })

# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n")
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

for r in all_results:

    print(
        f"{r['subject']} | "
        f"ACC={r['accuracy']:.4f}"
    )

    print(
        f"  12Hz trials : "
        f"S12={r['s12_12']:.6f} "
        f"S15={r['s15_12']:.6f}"
    )

    print(
        f"  15Hz trials : "
        f"S12={r['s12_15']:.6f} "
        f"S15={r['s15_15']:.6f}"
    )

mean_acc = np.mean(
    [
        r["accuracy"]
        for r in all_results
    ]
)

print("\nMean Accuracy:")
print(f"{mean_acc:.4f}")

print("\nDONE")