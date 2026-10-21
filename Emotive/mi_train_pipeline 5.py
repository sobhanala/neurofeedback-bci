# mi_train_pipeline_v5.py

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
MI_CHANNELS = [
    "F3","FC5","AF3","F7",
    "T7","P7","O1","O2",
    "P8","T8","F8","AF4",
    "FC6","F4"
]
BANDPASS = (8, 30)
EPOCH_LENGTH = 2   # seconds
SFREQ = 256.0

# Subject-specific best offsets from sweep
BEST_OFFSETS = {
    "P100": 1.25,  # s after imagery_on
    "P101": 0.0
}

MODEL_ROOT = r"mi_model_v5"
os.makedirs(MODEL_ROOT, exist_ok=True)
# ==========================================

# Discover subjects
SUBJECTS = sorted([d.replace("sub-", "") for d in os.listdir(BIDS_ROOT) if d.startswith("sub-")])
FINAL_RESULTS = []
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
        raw.filter(BANDPASS[0], BANDPASS[1], fir_design='firwin', verbose=False)

        df_ann = raw.annotations.to_data_frame().rename(columns={'description':'trial_type'})
        first_onset = df_ann['onset'].iloc[0]

        # Guided trials
        guided_events = df_ann[df_ann['trial_type'].str.contains('imagery_on')]
        # Free-choice trials
        free_events = df_ann[df_ann['trial_type'].str.contains('imagery_start')]

        # Combine events
        combined_events = pd.concat([guided_events, free_events], ignore_index=True)

        for idx, row in combined_events.iterrows():
            onset = (row['onset'] - first_onset).total_seconds()
            start = onset + offset
            stop  = start + EPOCH_LENGTH
            start_samp = int(start * SFREQ)
            stop_samp = int(stop * SFREQ)
            
            # Only extract and append if it's left or right imagery
            if ('imagery_on' in row['trial_type']) or ('imagery_start' in row['trial_type']):
                if ('left' in row['trial_type']) or ('direction=left' in row['trial_type']):
                    epoch_data = raw.get_data(start=start_samp, stop=stop_samp)
                    epoch_data = (epoch_data - epoch_data.mean(axis=1, keepdims=True)) / epoch_data.std(axis=1, keepdims=True)
                    X_all.append(epoch_data)
                    y_all.append(0)
                    groups.append(ses)
                elif ('right' in row['trial_type']) or ('direction=right' in row['trial_type']):
                    epoch_data = raw.get_data(start=start_samp, stop=stop_samp)
                    epoch_data = (epoch_data - epoch_data.mean(axis=1, keepdims=True)) / epoch_data.std(axis=1, keepdims=True)
                    X_all.append(epoch_data)
                    y_all.append(1)
                    groups.append(ses)
                else:
                    # skip any unlabeled or invalid row
                    continue


    X_all = np.array(X_all)
    y_all = np.array(y_all)
    groups = np.array(groups)
    print(f"Total epochs: {X_all.shape[0]} | Left: {np.sum(y_all==0)} | Right: {np.sum(y_all==1)}")

    # CSP + LDA
    csp = CSP(n_components=6, reg='ledoit_wolf', log=True)
    print("X shape:", X_all.shape)
    print("y shape:", y_all.shape)
    print("Left:", np.sum(y_all==0), "Right:", np.sum(y_all==1))
    X_csp = csp.fit_transform(X_all, y_all)
    lda = LDA(shrinkage='auto', solver='lsqr')

    # LOSO-CV
    logo = LeaveOneGroupOut()
    scores = cross_val_score(lda, X_csp, y_all, cv=logo.split(X_csp, y_all, groups))
    print(f"LOSO-CV Accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    FINAL_RESULTS.append({
        "subject": sub,
        "accuracy": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "channels": len(MI_CHANNELS)
    })
    # Fit final model
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
        "epoch_offset_s": offset,
        "epoch_length_s": EPOCH_LENGTH,
        "csp_components": 6,
        "LOSO_CV_mean": float(np.mean(scores)),
        "LOSO_CV_std": float(np.std(scores))
    }
    with open(os.path.join(sub_model_root, 'model_id.json'), 'w') as f:
        json.dump(model_id, f, indent=4)

    print(f"Saved final model artifacts for subject {sub} at offset {offset}s")

        # ================== ADD THIS BLOCK HERE ==================
    # Compute and print final LOSO-CV for the subject
    X_csp_final = csp.transform(X_all)  # transform all epochs
    logo = LeaveOneGroupOut()
    scores_final = cross_val_score(lda, X_csp_final, y_all, cv=logo.split(X_csp_final, y_all, groups))
    print(f"Subject {sub} final LOSO-CV Accuracy: {np.mean(scores_final):.3f} ± {np.std(scores_final):.3f}")
    # =========================================================

    print("\n")
    print("="*50)
    print("FINAL RESULTS")
    print("="*50)

    for r in FINAL_RESULTS:
        print(
            f"{r['subject']} | "
            f"Channels={r['channels']} | "
            f"Accuracy={r['accuracy']:.3f} ± {r['std']:.3f}"
        )