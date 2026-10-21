import os
import numpy as np

from scipy.signal import welch
from sklearn.metrics import accuracy_score, confusion_matrix

# ==================================================
# CONFIG
# ==================================================

DATASET_ROOT = "ssvep_dataset"

SUBJECTS = [
    "P100",
    "P101"
]

FS = 256

FREQ_1 = 12
FREQ_2 = 15

# O1 and O2 indices in your exported channel order
O1_IDX = 6
O2_IDX = 7

# ==================================================
# HELPER
# ==================================================

def get_power_at(freqs, psd, target_freq):

    idx = np.argmin(
        np.abs(freqs - target_freq)
    )

    return psd[idx]

# ==================================================
# EVALUATION
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

    predictions = []

    power12_trials = []
    power15_trials = []

    class12_power12 = []
    class12_power15 = []

    class15_power12 = []
    class15_power15 = []

    for epoch, label in zip(X, y):

        # --------------------------------------
        # Use only O1 and O2
        # --------------------------------------

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

        p12 = get_power_at(
            freqs,
            psd,
            12
        )

        p15 = get_power_at(
            freqs,
            psd,
            15
        )

        power12_trials.append(p12)
        power15_trials.append(p15)

        pred = (
            0
            if p12 > p15
            else 1
        )

        predictions.append(pred)

        if label == 0:

            class12_power12.append(p12)
            class12_power15.append(p15)

        else:

            class15_power12.append(p12)
            class15_power15.append(p15)

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

    print("\nAccuracy")
    print(f"{acc:.4f}")

    print("\nConfusion Matrix")
    print(cm)

    print("\nMean Powers")

    print(
        f"12Hz trials -> "
        f"P12={np.mean(class12_power12):.6f} "
        f"P15={np.mean(class12_power15):.6f}"
    )

    print(
        f"15Hz trials -> "
        f"P12={np.mean(class15_power12):.6f} "
        f"P15={np.mean(class15_power15):.6f}"
    )

    all_results.append({
        "subject": subject,
        "accuracy": acc,
        "p12_12": np.mean(class12_power12),
        "p15_12": np.mean(class12_power15),
        "p12_15": np.mean(class15_power12),
        "p15_15": np.mean(class15_power15)
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
        f"P12={r['p12_12']:.6f} "
        f"P15={r['p15_12']:.6f}"
    )

    print(
        f"  15Hz trials : "
        f"P12={r['p12_15']:.6f} "
        f"P15={r['p15_15']:.6f}"
    )

mean_acc = np.mean(
    [r["accuracy"] for r in all_results]
)

print("\nMean Accuracy")
print(f"{mean_acc:.4f}")

print("\nDONE")