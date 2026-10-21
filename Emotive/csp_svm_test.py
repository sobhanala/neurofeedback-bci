import numpy as np

from mne.decoding import CSP

from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

DATASET_ROOT = "dataset"

SUBJECTS = [
    "P100",
    "P101"
]

print("\n" + "="*60)
print("CSP + SVM TEST")
print("="*60)

for sub in SUBJECTS:

    print(f"\n===== SUBJECT {sub} =====")

    X = np.load(
        f"{DATASET_ROOT}/subject_{sub}/X.npy"
    )

    y = np.load(
        f"{DATASET_ROOT}/subject_{sub}/y.npy"
    )

    groups = np.load(
        f"{DATASET_ROOT}/subject_{sub}/groups.npy"
    )

    # Same CSP that won previously
    csp = CSP(
        n_components=6,
        reg='ledoit_wolf',
        log=True
    )

    X_csp = csp.fit_transform(X, y)

    logo = LeaveOneGroupOut()

    # -------------------------
    # Linear SVM
    # -------------------------

    svm_linear = SVC(
        kernel='linear'
    )

    scores_linear = cross_val_score(
        svm_linear,
        X_csp,
        y,
        cv=logo.split(X_csp, y, groups)
    )

    print(
        f"Linear SVM : "
        f"{np.mean(scores_linear):.3f} ± "
        f"{np.std(scores_linear):.3f}"
    )

    # -------------------------
    # RBF SVM
    # -------------------------

    svm_rbf = SVC(
        kernel='rbf',
        gamma='scale'
    )

    scores_rbf = cross_val_score(
        svm_rbf,
        X_csp,
        y,
        cv=logo.split(X_csp, y, groups)
    )

    print(
        f"RBF SVM    : "
        f"{np.mean(scores_rbf):.3f} ± "
        f"{np.std(scores_rbf):.3f}"
    )

print("\n" + "="*60)
print("DONE")
print("="*60)