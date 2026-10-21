# inspect_ssvep_frequencies.py

import os
import numpy as np
from scipy.signal import welch

# ==================================================
# CONFIG
# ==================================================

DATASET_ROOT = "ssvep_dataset_raw"

SUBJECTS = [
    "P100",
    "P101"
]

FS = 256

O1_IDX = 6
O2_IDX = 7

TARGET_FREQS = [
    12,
    15,
    24,
    30,
    36,
    45
]

# ==================================================
# HELPER
# ==================================================

def get_power(freqs, psd, target_freq):

    idx = np.argmin(
        np.abs(freqs - target_freq)
    )

    return psd[idx]

# ==================================================
# MAIN
# ==================================================

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

    powers = {
        f: []
        for f in TARGET_FREQS
    }

    for epoch in X:

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

        for f in TARGET_FREQS:

            powers[f].append(
                get_power(
                    freqs,
                    psd,
                    f
                )
            )

    print("\nAverage PSD:")

    for f in TARGET_FREQS:

        print(
            f"{f:>2} Hz : "
            f"{np.mean(powers[f]):.6f}"
        )

print("\nDONE")