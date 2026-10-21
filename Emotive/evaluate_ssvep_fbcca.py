import os
import numpy as np

from scipy.signal import butter, filtfilt
from sklearn.cross_decomposition import CCA
from sklearn.metrics import accuracy_score, confusion_matrix

# ==================================================
# CONFIG
# ==================================================

DATASET_ROOT = "ssvep_dataset_raw"

SUBJECTS = [
    "P100",
    "P101"
]

FS = 256

TARGET_FREQS = [
    12,
    15
]

N_HARMONICS = 3

# FBCCA filter bank
FILTER_BANKS = [
    (8, 15),
    (15, 22),
    (22, 30),
    (30, 45)
]

# ==================================================
# FILTER
# ==================================================

def bandpass(data, low, high, fs):

    nyq = fs / 2

    b, a = butter(
        4,
        [low / nyq, high / nyq],
        btype="band"
    )

    return filtfilt(
        b,
        a,
        data,
        axis=1
    )

# ==================================================
# REFERENCE SIGNALS
# ==================================================

def build_reference(freq, n_samples):

    t = np.arange(n_samples) / FS

    refs = []

    for h in range(1, N_HARMONICS + 1):

        refs.append(
            np.sin(
                2 * np.pi * h * freq * t
            )
        )

        refs.append(
            np.cos(
                2 * np.pi * h * freq * t
            )
        )

    return np.array(refs)

# ==================================================
# CCA CORRELATION
# ==================================================

def cca_correlation(epoch, freq):

    refs = build_reference(
        freq,
        epoch.shape[1]
    )

    X = epoch.T
    Y = refs.T

    cca = CCA(
        n_components=1,
        max_iter=1000
    )

    try:

        cca.fit(X, Y)

        Xc, Yc = cca.transform(X, Y)

        corr = np.corrcoef(
            Xc[:, 0],
            Yc[:, 0]
        )[0, 1]

        if np.isnan(corr):
            return 0.0

        return abs(corr)

    except Exception:

        return 0.0

# ==================================================
# FBCCA SCORE
# ==================================================

def fbcca_score(epoch, freq):

    score = 0.0

    for i, (low, high) in enumerate(FILTER_BANKS):

        filtered = bandpass(
            epoch,
            low,
            high,
            FS
        )

        corr = cca_correlation(
            filtered,
            freq
        )

        weight = 1.0 / ((i + 1) ** 1.25)

        score += weight * (corr ** 2)

    return score

# ==================================================
# SUBJECT EVALUATION
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

    for epoch in X:

        scores = []

        for freq in TARGET_FREQS:

            score = fbcca_score(
                epoch,
                freq
            )

            scores.append(score)

        print(scores)
        pred = np.argmax(scores)

        predictions.append(pred)

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

    n12 = np.sum(y == 0)
    n15 = np.sum(y == 1)

    correct12 = np.sum(
        (y == 0)
        &
        (predictions == 0)
    )

    correct15 = np.sum(
        (y == 1)
        &
        (predictions == 1)
    )

    acc12 = correct12 / n12
    acc15 = correct15 / n15

    print("\nAccuracy:")
    print(f"{acc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClass Accuracy:")
    print(f"12Hz : {acc12:.4f}")
    print(f"15Hz : {acc15:.4f}")

    all_results.append({
        "subject": subject,
        "accuracy": acc,
        "acc12": acc12,
        "acc15": acc15
    })

# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n")
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

for result in all_results:

    print(
        f"{result['subject']} | "
        f"ACC={result['accuracy']:.4f} | "
        f"12Hz={result['acc12']:.4f} | "
        f"15Hz={result['acc15']:.4f}"
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