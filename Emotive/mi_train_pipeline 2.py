# mi_train_pipeline_v3.py

import os
import mne
from mne_bids import BIDSPath, read_raw_bids
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
import numpy as np
import pickle
import json

# ================= CONFIG ==================
BIDS_ROOT = r"MI_BIDS"
TASK = "mi"

# Use all MI channels plus some frontal/temporal for CSP
MI_CHANNELS = ["F3","FC5","AF3","F7","T7","P7","T8","F8","AF4","FC6","F4"]
BANDPASS = (8, 30)

# Epoch window (experiment with different offsets)
EPOCH_START_OFFSET = 0.25  # seconds after imagery_on
EPOCH_LENGTH = 2.0          # seconds

SFREQ = 256.0
MODEL_ROOT = r"mi_model_v3"
os.makedirs(MODEL_ROOT, exist_ok=True)
# ==========================================

# Discover subjects
SUBJECTS = sorted([d.replace("sub-", "") for d in os.listdir(BIDS_ROOT) if d.startswith("sub-")])

for sub in SUBJECTS:
    print(f"\n=== Training subject {sub} ===")
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
        raw.filter(BANDPASS[0], BANDPASS[1], fir_design='firwin', verbose=False)

        df_ann = raw.annotations.to_data_frame().rename(columns={'description':'trial_type'})

        # Select guided trials
        guided_events = df_ann[df_ann['trial_type'].str.contains('imagery_on')]
        first_onset = guided_events['onset'].iloc[0]

        for idx, row in guided_events.iterrows():
            onset = (row['onset'] - first_onset).total_seconds()
            start = onset + EPOCH_START_OFFSET
            stop  = start + EPOCH_LENGTH
            start_samp = int(start * SFREQ)
            stop_samp = int(stop * SFREQ)
            epoch_data = raw.get_data(start=start_samp, stop=stop_samp)

            # Channel-wise z-score normalization
            epoch_data = (epoch_data - epoch_data.mean(axis=1, keepdims=True)) / epoch_data.std(axis=1, keepdims=True)

            X_all.append(epoch_data)
            y_all.append(0 if 'left' in row['trial_type'] else 1)
            groups.append(ses)

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    groups = np.array(groups)

    print(f"Total epochs: {X_all.shape[0]} | Left: {np.sum(y_all==0)} | Right: {np.sum(y_all==1)}")

    # CSP + LDA
    csp = CSP(n_components=6, reg='ledoit_wolf', log=True)
    X_csp = csp.fit_transform(X_all, y_all)

    lda = LDA(shrinkage='auto', solver='lsqr')
    logo = LeaveOneGroupOut()
    scores = cross_val_score(lda, X_csp, y_all, cv=logo.split(X_csp, y_all, groups))
    print(f"LOSO-CV Accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    # Fit final model on all data
    lda.fit(X_csp, y_all)

    # Save model artifacts
    sub_model_root = os.path.join(MODEL_ROOT, f"subject_{sub}")
    os.makedirs(sub_model_root, exist_ok=True)

    with open(os.path.join(sub_model_root, 'csp.pkl'), 'wb') as f:
        pickle.dump(csp, f)
    with open(os.path.join(sub_model_root, 'lda.pkl'), 'wb') as f:
        pickle.dump(lda, f)

    model_id = {
        "subject": sub,
        "channels": MI_CHANNELS,
        "bandpass": BANDPASS,
        "epoch_start_offset": EPOCH_START_OFFSET,
        "epoch_length_s": EPOCH_LENGTH,
        "csp_components": 6,
        "LOSO_CV_mean": float(np.mean(scores)),
        "LOSO_CV_std": float(np.std(scores))
    }
    with open(os.path.join(sub_model_root, 'model_id.json'), 'w') as f:
        json.dump(model_id, f, indent=4)

    print(f"Saved model artifacts for subject {sub} in {sub_model_root}")