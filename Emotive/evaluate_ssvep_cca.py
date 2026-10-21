import os
import numpy as np

from sklearn.cross_decomposition import CCA
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

FREQUENCIES = [
    12,
    15
]

N_HARMONICS = 3

# ==================================================
# REFERENCE GENERATION
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
# CCA SCORE
# ==================================================

def cca_score(epoch, freq):

    refs = build_reference(
        freq,
        epoch.shape[1]
    )

    X = epoch.T
    Y = refs.T

    cca = CCA(
        n_components=1
    )

    cca.fit(X, Y)

    X_c, Y_c = cca.transform(X, Y)

    corr = np.corrcoef(
        X_c[:, 0],
        Y_c[:, 0]
    )[0, 1]

    return abs(corr)


# ==================================================
# EVALUATE SUBJECT
# ==================================================

all_subject_results = []

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

        for freq in FREQUENCIES:

            score = cca_score(
                epoch,
                freq
            )

            scores.append(score)

        print(scores)
        pred = np.argmax(scores)

        predictions.append(pred)

    predictions = np.array(predictions)

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

    print("\nClass Accuracy:")
    print(f"12Hz : {acc12:.4f}")
    print(f"15Hz : {acc15:.4f}")

    all_subject_results.append({
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

for result in all_subject_results:

    print(
        f"{result['subject']} | "
        f"ACC={result['accuracy']:.4f} | "
        f"12Hz={result['acc12']:.4f} | "
        f"15Hz={result['acc15']:.4f}"
    )

mean_acc = np.mean(
    [
        r["accuracy"]
        for r in all_subject_results
    ]
)

print("\nMean Accuracy:")
print(f"{mean_acc:.4f}")

print("\nDONE")