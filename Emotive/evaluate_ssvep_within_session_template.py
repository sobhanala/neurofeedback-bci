import os
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

# ==================================================
# CONFIG
# ==================================================

DATASET_ROOT = "ssvep_dataset_raw"

SUBJECTS = [
    "P100",
    "P101"
]

N_SPLITS = 20
TEST_SIZE = 0.20

# ==================================================
# CORRELATION
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

subject_results = []

for subject in SUBJECTS:

    print("\n" + "=" * 60)
    print(f"SUBJECT {subject}")
    print("=" * 60)

    subject_path = os.path.join(
        DATASET_ROOT,
        f"subject_{subject}"
    )

    X = np.load(
        os.path.join(subject_path, "X.npy")
    )

    y = np.load(
        os.path.join(subject_path, "y_freq.npy")
    )

    groups = np.load(
        os.path.join(subject_path, "groups.npy")
    )

    sessions = np.unique(groups)

    session_accs = []

    for session in sessions:

        idx = groups == session

        X_ses = X[idx]
        y_ses = y[idx]

        splitter = StratifiedShuffleSplit(
            n_splits=N_SPLITS,
            test_size=TEST_SIZE,
            random_state=42
        )

        fold_accs = []

        for train_idx, test_idx in splitter.split(X_ses, y_ses):

            X_train = X_ses[train_idx]
            y_train = y_ses[train_idx]

            X_test = X_ses[test_idx]
            y_test = y_ses[test_idx]

            # -----------------------------------------
            # Templates
            # -----------------------------------------

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

            acc = accuracy_score(
                y_test,
                preds
            )

            fold_accs.append(acc)

        mean_session_acc = np.mean(fold_accs)

        session_accs.append(mean_session_acc)

        print(
            f"{session} : "
            f"{mean_session_acc:.4f}"
        )

    subject_mean = np.mean(session_accs)

    print("\nSubject Mean:")
    print(f"{subject_mean:.4f}")

    subject_results.append(subject_mean)

# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n")
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

for subject, acc in zip(
    SUBJECTS,
    subject_results
):

    print(
        f"{subject} : "
        f"{acc:.4f}"
    )

print("\nOverall Mean:")

print(
    f"{np.mean(subject_results):.4f}"
)

print("\nDONE")