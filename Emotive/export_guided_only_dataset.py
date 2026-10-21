import os
import numpy as np
import pandas as pd

from mne_bids import BIDSPath, read_raw_bids

# ======================================
# CONFIG
# ======================================

BIDS_ROOT = "MI_BIDS"

TASK = "mi"

MI_CHANNELS = [
    "F3","FC5","AF3","F7",
    "T7","P7","O1","O2",
    "P8","T8","F8","AF4",
    "FC6","F4"
]

BANDPASS = (8, 30)

EPOCH_LENGTH = 2.0
SFREQ = 256.0

BEST_OFFSETS = {
    "P100": 1.25,
    "P101": 0.0,
    "P200": 0.0,
    "P201": 0.0
}

OUT_ROOT = "guided_dataset"

os.makedirs(OUT_ROOT, exist_ok=True)

# ======================================
# SUBJECTS
# ======================================

SUBJECTS = sorted([
    d.replace("sub-", "")
    for d in os.listdir(BIDS_ROOT)
    if d.startswith("sub-")
])

# ======================================
# EXPORT
# ======================================

for sub in SUBJECTS:

    print(f"\n=== Exporting Guided-Only {sub} ===")

    offset = BEST_OFFSETS[sub]

    sub_folder = os.path.join(
        BIDS_ROOT,
        f"sub-{sub}"
    )

    sessions = sorted([
        d.replace("ses-", "")
        for d in os.listdir(sub_folder)
        if d.startswith("ses-")
    ])

    X_all = []
    y_all = []
    groups = []

    for ses in sessions:

        bids_path = BIDSPath(
            subject=sub,
            session=ses,
            task=TASK,
            root=BIDS_ROOT,
            datatype="eeg"
        )

        raw = read_raw_bids(
            bids_path=bids_path,
            verbose=False
        )

        raw.pick(MI_CHANNELS)

        raw.load_data()

        raw.filter(
            BANDPASS[0],
            BANDPASS[1],
            fir_design="firwin",
            verbose=False
        )

        df_ann = raw.annotations.to_data_frame()
        df_ann = df_ann.rename(
            columns={"description":"trial_type"}
        )

        first_onset = df_ann["onset"].iloc[0]

        # ======================================
        # GUIDED ONLY
        # Keep only imagery_on events
        # ======================================

        guided_events = df_ann[
            df_ann["trial_type"].str.contains(
                "imagery_on",
                na=False
            )
        ]

        for _, row in guided_events.iterrows():

            onset = (
                row["onset"] - first_onset
            ).total_seconds()

            start = onset + offset
            stop = start + EPOCH_LENGTH

            start_samp = int(start * SFREQ)
            stop_samp = int(stop * SFREQ)

            epoch = raw.get_data(
                start=start_samp,
                stop=stop_samp
            )

            if epoch.shape[1] != int(EPOCH_LENGTH * SFREQ):
                continue

            epoch = (
                epoch - epoch.mean(
                    axis=1,
                    keepdims=True
                )
            ) / (
                epoch.std(
                    axis=1,
                    keepdims=True
                ) + 1e-8
            )

            if "left" in row["trial_type"]:
                label = 0

            elif "right" in row["trial_type"]:
                label = 1

            else:
                continue

            X_all.append(epoch)
            y_all.append(label)
            groups.append(ses)

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    groups = np.array(groups)

    print(
        f"X shape: {X_all.shape}"
    )

    print(
        f"Left: {(y_all==0).sum()} "
        f"Right: {(y_all==1).sum()}"
    )

    subject_out = os.path.join(
        OUT_ROOT,
        f"subject_{sub}"
    )

    os.makedirs(
        subject_out,
        exist_ok=True
    )

    np.save(
        os.path.join(subject_out, "X.npy"),
        X_all
    )

    np.save(
        os.path.join(subject_out, "y.npy"),
        y_all
    )

    np.save(
        os.path.join(subject_out, "groups.npy"),
        groups
    )

    print(
        f"Saved {subject_out}"
    )

print("\nDONE")