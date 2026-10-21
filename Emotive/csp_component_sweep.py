import numpy as np

from mne.decoding import CSP

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

DATASET_ROOT = "dataset"

SUBJECTS = [
    "P200",
    "P201"
]

COMPONENTS_TO_TEST = [
    2,
    4,
    6,
    8,
    10,
    12
]

print("\n" + "="*60)
print("CSP COMPONENT SWEEP")
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

    best_acc = 0
    best_comp = None

    for n_comp in COMPONENTS_TO_TEST:

        csp = CSP(
            n_components=n_comp,
            reg='ledoit_wolf',
            log=True
        )

        X_csp = csp.fit_transform(X, y)

        lda = LDA(
            shrinkage='auto',
            solver='lsqr'
        )

        logo = LeaveOneGroupOut()

        scores = cross_val_score(
            lda,
            X_csp,
            y,
            cv=logo.split(X_csp, y, groups)
        )

        mean_acc = np.mean(scores)
        std_acc = np.std(scores)

        print(
            f"CSP={n_comp:2d} | "
            f"Accuracy={mean_acc:.3f} ± {std_acc:.3f}"
        )

        if mean_acc > best_acc:
            best_acc = mean_acc
            best_comp = n_comp

    print(
        f"\nBEST FOR {sub}: "
        f"CSP={best_comp} | "
        f"Accuracy={best_acc:.3f}"
    )

print("\n" + "="*60)
print("SWEEP COMPLETE")
print("="*60)