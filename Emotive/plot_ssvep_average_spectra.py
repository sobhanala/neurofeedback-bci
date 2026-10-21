# plot_ssvep_average_spectra.py

import os
import numpy as np
import matplotlib.pyplot as plt

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

# O1 / O2 indices
O1_IDX = 6
O2_IDX = 7

FREQ_MARKERS = [
    12,
    15,
    24,
    30,
    36,
    45
]

# ==================================================
# MAIN
# ==================================================

for subject in SUBJECTS:

    print(f"\nProcessing {subject}")

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

    psd_12_list = []
    psd_15_list = []

    freqs_ref = None

    for epoch, label in zip(X, y):

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

        if freqs_ref is None:
            freqs_ref = freqs

        if label == 0:

            psd_12_list.append(psd)

        else:

            psd_15_list.append(psd)

    mean_psd_12 = np.mean(
        psd_12_list,
        axis=0
    )

    mean_psd_15 = np.mean(
        psd_15_list,
        axis=0
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        freqs_ref,
        mean_psd_12,
        label="Attend 12 Hz"
    )

    plt.plot(
        freqs_ref,
        mean_psd_15,
        label="Attend 15 Hz"
    )

    for f in FREQ_MARKERS:

        plt.axvline(
            f,
            linestyle="--",
            alpha=0.5
        )

    plt.xlim(0, 50)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")

    plt.title(
        f"{subject} - Average O1/O2 Spectrum"
    )

    plt.legend()

    plt.grid(True)

    save_name = (
        f"{subject}_average_spectrum.png"
    )

    plt.savefig(
        save_name,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"Saved: {save_name}"
    )

    plt.show()

print("\nDONE")