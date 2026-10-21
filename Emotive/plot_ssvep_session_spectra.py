import os
import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# CONFIG
# ==================================================

DATASET_ROOT = "ssvep_dataset_raw"

SUBJECT = "P101"      # change later if needed

FS = 256

O1_IDX = 6
O2_IDX = 7

# ==================================================
# LOAD DATA
# ==================================================

subject_path = os.path.join(
    DATASET_ROOT,
    f"subject_{SUBJECT}"
)

X = np.load(
    os.path.join(subject_path, "X.npy")
)

groups = np.load(
    os.path.join(subject_path, "groups.npy")
)

sessions = np.unique(groups)

print("Sessions found:")
print(sessions)

# ==================================================
# FFT HELPER
# ==================================================

def average_spectrum(trials):

    psds = []

    for epoch in trials:

        eeg = epoch[
            [O1_IDX, O2_IDX],
            :
        ]

        eeg = eeg.mean(axis=0)

        fft_vals = np.abs(
            np.fft.rfft(eeg)
        ) ** 2

        freqs = np.fft.rfftfreq(
            len(eeg),
            d=1.0 / FS
        )

        psds.append(fft_vals)

    return freqs, np.mean(psds, axis=0)

# ==================================================
# PLOT
# ==================================================

plt.figure(figsize=(12, 6))

for ses in sessions:

    idx = np.where(
        groups == ses
    )[0]

    freqs, spectrum = average_spectrum(
        X[idx]
    )

    plt.plot(
        freqs,
        spectrum,
        label=str(ses)
    )

plt.xlim(0, 50)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")

plt.title(
    f"{SUBJECT} Session Spectra (O1/O2)"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    f"{SUBJECT}_session_spectra.png",
    dpi=300
)

plt.show()

print("\nDONE")