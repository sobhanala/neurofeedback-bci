import numpy as np

from mne.decoding import CSP

from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

DATASET_ROOT = "guided_dataset"

SUBJECTS = [
    "P100",
    "P101"
]

print("\n" + "="*60)
print("GUIDED ONLY : CSP + LINEAR SVM")
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

    csp = CSP(
        n_components=6,
        reg='ledoit_wolf',
        log=True
    )

    X_csp = csp.fit_transform(X, y)

    svm = SVC(
        kernel='linear'
    )

    logo = LeaveOneGroupOut()

    scores = cross_val_score(
        svm,
        X_csp,
        y,
        cv=logo.split(X_csp, y, groups)
    )

    print(
        f"Accuracy: "
        f"{np.mean(scores):.3f} ± "
        f"{np.std(scores):.3f}"
    )

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)