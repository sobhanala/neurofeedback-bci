import numpy as np

from pyriemann.estimation import Covariances
from pyriemann.classification import MDM

from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

DATASET_ROOT = "dataset"

SUBJECTS = [
    "P100",
    "P101"
]

print("\n" + "="*50)
print("RIEMANNIAN CLASSIFICATION")
print("="*50)

for sub in SUBJECTS:

    X = np.load(
        f"{DATASET_ROOT}/subject_{sub}/X.npy"
    )

    y = np.load(
        f"{DATASET_ROOT}/subject_{sub}/y.npy"
    )

    groups = np.load(
        f"{DATASET_ROOT}/subject_{sub}/groups.npy"
    )

    print(f"\nSubject {sub}")
    print("X:", X.shape)

    cov = Covariances().fit_transform(X)

    clf = MDM()

    logo = LeaveOneGroupOut()

    scores = cross_val_score(
        clf,
        cov,
        y,
        cv=logo.split(cov, y, groups)
    )

    print(
        f"Accuracy: "
        f"{np.mean(scores):.3f} ± {np.std(scores):.3f}"
    )

print("\n" + "="*50)
print("DONE")
print("="*50)