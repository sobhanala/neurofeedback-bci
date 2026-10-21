import os
import numpy as np

from scipy.linalg import eigh
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
# TRCA
# ==================================================

def train_trca(X_class):

    """
    X_class:
        (n_trials, n_channels, n_samples)
    """

    n_trials, n_channels, n_samples = X_class.shape

    S = np.zeros((n_channels, n_channels))
    Q = np.zeros((n_channels, n_channels))

    # ------------------------------------------------
    # Q matrix
    # ------------------------------------------------

    for trial in X_class:

        Q += trial @ trial.T

    # ------------------------------------------------
    # S matrix
    # ------------------------------------------------

    for i in range(n_trials):

        Xi = X_class[i]

        Xi = Xi - Xi.mean(
            axis=1,
            keepdims=True
        )

        for j in range(i + 1, n_trials):

            Xj = X_class[j]

            Xj = Xj - Xj.mean(
                axis=1,
                keepdims=True
            )

            S += Xi @ Xj.T
            S += Xj @ Xi.T

    # ------------------------------------------------
    # Generalized eigenvalue problem
    # ------------------------------------------------

    Q += 1e-6 * np.eye(n_channels)

    eigvals, eigvecs = eigh(
        S,
        Q
    )

    w = eigvecs[:, -1]

    return w


# ==================================================
# TEMPLATE
# ==================================================

def make_template(X_class, w):

    projected = []

    for trial in X_class:

        projected.append(
            w.T @ trial
        )

    return np.mean(
        projected,
        axis=0
    )


# ==================================================
# CORRELATION
# ==================================================

def corr(a, b):

    a = a.flatten()
    b = b.flatten()

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

        # --------------------------------------------
        # Train TRCA filters
        # --------------------------------------------

        X12 = X_train[y_train == 0]
        X15 = X_train[y_train == 1]

        w12 = train_trca(X12)
        w15 = train_trca(X15)

        template12 = make_template(
            X12,
            w12
        )

        template15 = make_template(
            X15,
            w15
        )

        preds = []

        for trial in X_test:

            proj12 = w12.T @ trial
            proj15 = w15.T @ trial

            r12 = corr(
                proj12,
                template12
            )

            r15 = corr(
                proj15,
                template15
            )

            pred = (
                0
                if r12 > r15
                else 1
            )

            preds.append(pred)

        preds = np.array(preds)

        fold_acc = accuracy_score(
            y_test,
            preds
        )

        print(
            f"{test_session}: "
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

    acc12 = np.mean(
        all_preds[all_true == 0] == 0
    )

    acc15 = np.mean(
        all_preds[all_true == 1] == 1
    )

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
    [r["accuracy"] for r in all_results]
)

print("\nMean Accuracy:")
print(f"{mean_acc:.4f}")

print("\nDONE")