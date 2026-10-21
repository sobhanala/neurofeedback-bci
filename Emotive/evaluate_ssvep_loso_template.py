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

    groups = np.load(
        os.path.join(
            subject_path,
            "groups.npy"
        )
    )

    sessions = np.unique(groups)

    all_preds = []
    all_true = []

    print("\nPer-session results")

    for test_session in sessions:

        train_mask = groups != test_session
        test_mask = groups == test_session

        X_train = X[train_mask]
        y_train = y[train_mask]

        X_test = X[test_mask]
        y_test = y[test_mask]

        # ----------------------------------------
        # Build templates from TRAIN only
        # ----------------------------------------

        template12 = np.mean(
            X_train[y_train == 0],
            axis=0
        )

        template15 = np.mean(
            X_train[y_train == 1],
            axis=0
        )

        preds = []

        for epoch in X_test:

            s12 = corr_score(
                epoch,
                template12
            )

            s15 = corr_score(
                epoch,
                template15
            )

            pred = (
                0
                if s12 > s15
                else 1
            )

            preds.append(pred)

        preds = np.array(preds)

        fold_acc = accuracy_score(
            y_test,
            preds
        )

        print(
            f"{test_session} : "
            f"{fold_acc:.4f}"
        )

        all_preds.extend(preds)
        all_true.extend(y_test)

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    acc = accuracy_score(
        all_true,
        all_preds
    )

    cm = confusion_matrix(
        all_true,
        all_preds
    )

    n12 = np.sum(all_true == 0)
    n15 = np.sum(all_true == 1)

    acc12 = np.sum(
        (all_true == 0)
        &
        (all_preds == 0)
    ) / n12

    acc15 = np.sum(
        (all_true == 1)
        &
        (all_preds == 1)
    ) / n15

    print("\nOverall Accuracy:")
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

for r in all_results:

    print(
        f"{r['subject']} | "
        f"ACC={r['accuracy']:.4f} | "
        f"12Hz={r['acc12']:.4f} | "
        f"15Hz={r['acc15']:.4f}"
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