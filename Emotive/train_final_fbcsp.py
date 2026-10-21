import os
import json
import pickle
import numpy as np

from scipy.signal import butter, filtfilt

from mne.decoding import CSP

from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif
DATASET_ROOT = "dataset"

MODEL_ROOT = "final_models"

os.makedirs(MODEL_ROOT, exist_ok=True)

SFREQ = 256
MI_CHANNELS = [
    "F3",
    "FC5",
    "AF3",
    "F7",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "F8",
    "AF4",
    "FC6",
    "F4"
]
BANDS = [
    (8,12),
    (12,16),
    (16,20),
    (20,24),
    (24,30)
]

BEST_CONFIG = {
    #'''
    #"P100": {
    #    "csp_components": 4,
    #    "k": 12
    #},
    #"P101": {
    #    "csp_components": 6,
    #    "k": 30
    #    asdefasdfsadf
    #}
    #'''

    "P200": {
        "csp_components": 6,
        "k": 30
    },
    "P201": {
        "csp_components": 6,
        "k": 30
    }
    
}

def bandpass_epoch(epoch, low, high, fs):

    nyq = fs / 2

    b, a = butter(
        4,
        [low/nyq, high/nyq],
        btype='band'
    )

    return filtfilt(
        b,
        a,
        epoch,
        axis=1
    )

for sub in ["P200", "P201"]:

    print("\n" + "="*60)
    print(f"TRAINING {sub}")
    print("="*60)

    X = np.load(
        f"{DATASET_ROOT}/subject_{sub}/X.npy"
    )

    y = np.load(
        f"{DATASET_ROOT}/subject_{sub}/y.npy"
    )

    cfg = BEST_CONFIG[sub]

    CSP_COMPONENTS = cfg["csp_components"]
    K_FEATURES = cfg["k"]

    feature_blocks = []
    csp_models = []

    for low, high in BANDS:

        print(f"Band {low}-{high} Hz")

        X_band = np.array([
            bandpass_epoch(
                epoch,
                low,
                high,
                SFREQ
            )
            for epoch in X
        ])

        csp = CSP(
            n_components=CSP_COMPONENTS,
            reg='ledoit_wolf',
            log=True
        )

        X_csp = csp.fit_transform(
            X_band,
            y
        )

        feature_blocks.append(
            X_csp
        )

        csp_models.append(
            csp
        )

    X_features = np.concatenate(
        feature_blocks,
        axis=1
    )

    selector = SelectKBest(
        mutual_info_classif,
        k=min(
            K_FEATURES,
            X_features.shape[1]
        )
    )

    X_selected = selector.fit_transform(
        X_features,
        y
    )

    svm = SVC(
        kernel="linear",
        probability=True
    )

    svm.fit(
        X_selected,
        y
    )

    save_dir = os.path.join(
        MODEL_ROOT,
        sub
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    with open(
        os.path.join(save_dir, "csp_models.pkl"),
        "wb"
    ) as f:
        pickle.dump(
            csp_models,
            f
        )

    with open(
        os.path.join(save_dir, "selector.pkl"),
        "wb"
    ) as f:
        pickle.dump(
            selector,
            f
        )

    with open(
        os.path.join(save_dir, "svm.pkl"),
        "wb"
    ) as f:
        pickle.dump(
            svm,
            f
        )

    metadata = {
        "subject": sub,

        "bands": BANDS,

        "csp_components": CSP_COMPONENTS,

        "k_features": K_FEATURES,

        "channels": MI_CHANNELS,

        "sfreq": 256,

        "epoch_length_seconds": 2.0,

        "epoch_length_samples": 512

        
    }
    with open(
        os.path.join(save_dir, "model_info.json"),
        "w"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4
        )

    print(
        f"Saved model to {save_dir}"
    )