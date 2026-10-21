# mi_train_pipeline_v6_fbcsp.py

import os
import mne
from mne_bids import BIDSPath, read_raw_bids
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
import numpy as np
import pickle
import json
import pandas as pd

# ================= CONFIG ==================
BIDS_ROOT = r"MI_BIDS"
TASK = "mi"

# All 14 Emotiv channels
MI_CHANNELS = ["F3","FC5","AF3","F7","T7","P7","O1","O2","P8","T8","F8","AF4","FC6","F4"]

# Subject-specific best offsets
BEST_OFFSETS = {
    "P100": 1.25,  # s after imagery_on
    "P101": 0.0
}

EPOCH_LENGTH = 2.0   # seconds
SFREQ = 256.0

# Frequency bands for FBCSP
BANDS = [
    (8,12),
    (12,16),
    (16,20),
    (20,24),
    (24,30)
]
CSP_COMPONENTS_PER_BAND = 2

MODEL_ROOT = r"mi_model_v6_fbcsp"
os.makedirs(MODEL_ROOT, exist_ok=True)
# ==========================================

# Discover subjects
SUBJECTS = sorted([d.replace("sub-", "") for d in os.listdir(BIDS_ROOT) if d.startswith("sub-")])

for sub in SUBJECTS:
    offset = BEST_OFFSETS[sub]
    print(f"\n=== Training subject {sub} | Best offset {offset}s ===")

    sub_folder = os.path.join(BIDS_ROOT, f"sub-{sub}")
    sessions = sorted([d.replace("ses-", "") for d in os.listdir(sub_folder) if d.startswith("ses-")])

    X_all = []
    y_all = []
    groups = []

    for ses in sessions:
        bids_path = BIDSPath(subject=sub, session=ses, task=TASK, root=BIDS_ROOT, datatype="eeg")
        raw = read_raw_bids(bids_path=bids_path, verbose=False)
        raw.pick(MI_CHANNELS)
        raw.load_data()

        df_ann = raw.annotations.to_data_frame().rename(columns={'description':'trial_type'})
        first_onset = df_ann['onset'].iloc[0]

        # Guided trials
        guided_events = df_ann[df_ann['trial_type'].str.contains('imagery_on')]
        # Free-choice trials
        free_events = df_ann[df_ann['trial_type'].str.contains('imagery_start')]

        # Combine events
        combined_events = pd.concat([guided_events, free_events], ignore_index=True)

        for idx, row in combined_events.iterrows():
            if not (('imagery_on' in row['trial_type']) or ('imagery_start' in row['trial_type'])):
                continue

            onset = (row['onset'] - first_onset).total_seconds()
            start = onset + offset
            stop = start + EPOCH_LENGTH
            start_samp = int(start * SFREQ)
            stop_samp = int(stop * SFREQ)
            epoch_data = raw.get_data(start=start_samp, stop=stop_samp)

            # Channel-wise z-score
            epoch_data = (epoch_data - epoch_data.mean(axis=1, keepdims=True)) / epoch_data.std(axis=1, keepdims=True)

            # Assign label
            if ('left' in row['trial_type']) or ('direction=left' in row['trial_type']):
                X_all.append(epoch_data)
                y_all.append(0)
                groups.append(ses)
            elif ('right' in row['trial_type']) or ('direction=right' in row['trial_type']):
                X_all.append(epoch_data)
                y_all.append(1)
                groups.append(ses)
            # skip invalid

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    groups = np.array(groups)
    print(f"Total epochs: {len(X_all)} | Left: {np.sum(y_all==0)} | Right: {np.sum(y_all==1)}")

    # ---------------- FBCSP ----------------
    X_features = []

    for band in BANDS:
        raw_band = raw.copy().filter(band[0], band[1], fir_design='firwin', verbose=False)
        band_epochs = []

        for epoch_data in X_all:
            # Extract band-limited signal for each epoch
            start_idx = 0
            stop_idx = epoch_data.shape[1]
            band_epochs.append(epoch_data[:, start_idx:stop_idx])
        band_epochs = np.array(band_epochs)

        # CSP for this band
        csp = CSP(n_components=CSP_COMPONENTS_PER_BAND, reg='ledoit_wolf', log=True)
        X_csp_band = csp.fit_transform(band_epochs, y_all)
        X_features.append(X_csp_band)

    # Concatenate CSP features from all bands
    X_csp_final = np.concatenate(X_features, axis=1)

    # ---------------- LDA ----------------
    lda = LDA(shrinkage='auto', solver='lsqr')
    logo = LeaveOneGroupOut()
    scores = cross_val_score(lda, X_csp_final, y_all, cv=logo.split(X_csp_final, y_all, groups))
    print(f"LOSO-CV Accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    lda.fit(X_csp_final, y_all)

    # ---------------- Save model ----------------
    sub_model_root = os.path.join(MODEL_ROOT, f"subject_{sub}")
    os.makedirs(sub_model_root, exist_ok=True)

    with open(os.path.join(sub_model_root, 'lda.pkl'), 'wb') as f:
        pickle.dump(lda, f)
    with open(os.path.join(sub_model_root, 'csp_features.pkl'), 'wb') as f:
        pickle.dump(X_features, f)

    model_id = {
        "subject": sub,
        "channels": MI_CHANNELS,
        "bands": BANDS,
        "epoch_offset_s": offset,
        "epoch_length_s": EPOCH_LENGTH,
        "CSP_components_per_band": CSP_COMPONENTS_PER_BAND,
        "LOSO_CV_mean": float(np.mean(scores)),
        "LOSO_CV_std": float(np.std(scores))
    }
    with open(os.path.join(sub_model_root, 'model_id.json'), 'w') as f:
        json.dump(model_id, f, indent=4)

    print(f"Saved FBCSP model for subject {sub}")