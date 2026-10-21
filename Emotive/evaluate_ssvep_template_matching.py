import os
import numpy as np

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

# ==================================================
# HELPERS
# ==================================================

def corr_score(a, b):

    a = a.reshape(-1)
    b = b.reshape(-1)

    a = a - np.mean(a)
    b = b - np.mean(b)

    denom = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denom < 1e-12:
        return 0.0

    return np.dot(a, b) / denom


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

    # ------------------------------------------------
    # Build class templates
    # ------------------------------------------------

    template12 = np.mean(
        X[y == 0],
        axis=0
    )

    template15 = np.mean(
        X[y == 1],
        axis=0
    )

    # ------------------------------------------------
    # Classify
    # ------------------------------------------------

    predictions = []

    scores12 = []
    scores15 = []

    for epoch in X:

        s12 = corr_score(
            epoch,
            template12
        )

        s15 = corr_score(
            epoch,
            template15
        )

        scores12.append(s12)
        scores15.append(s15)

        pred = (
            0
            if s12 > s15
            else 1
        )

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

    acc12 = np.sum(
        (y == 0)
        &
        (predictions == 0)
    ) / n12

    acc15 = np.sum(
        (y == 1)
        &
        (predictions == 1)
    ) / n15

    print("\nAccuracy:")
    print(f"{acc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClass Accuracy:")
    print(f"12Hz : {acc12:.4f}")
    print(f"15Hz : {acc15:.4f}")

    print("\nMean Scores:")
    print(
        f"Template12 = "
        f"{np.mean(scores12):.4f}"
    )
    print(
        f"Template15 = "
        f"{np.mean(scores15):.4f}"
    )

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

for r in all_results:

    print(
        f"{r['subject']} | "
        f"ACC={r['accuracy']:.4f} | "
        f"12Hz={r['acc12']:.4f} | "
        f"15Hz={r['acc15']:.4f}"
    )

mean_acc = np.mean(
    [r["accuracy"] for r in all_results]
)

print("\nMean Accuracy:")
print(f"{mean_acc:.4f}")

print("\nDONE")