import os
import re
import numpy as np
import pandas as pd

from mne_bids import BIDSPath, read_raw_bids

# ==========================
# CONFIG
# ==========================

BIDS_ROOT = r"SSVEP_BIDS"

TASK = "ssvep"

SSVEP_CHANNELS = [
    "F3","FC5","AF3","F7",
    "T7","P7","O1","O2",
    "P8","T8","F8","AF4",
    "FC6","F4"
]

BANDPASS = (8, 45)

SFREQ = 256.0

TRIAL_DURATION_SEC = 6.0
TRIAL_SAMPLES = int(TRIAL_DURATION_SEC * SFREQ)

OUT_ROOT = "ssvep_dataset"

os.makedirs(OUT_ROOT, exist_ok=True)

# ==========================
# SUBJECTS
# ==========================

SUBJECTS = sorted([
    d.replace("sub-", "")
    for d in os.listdir(BIDS_ROOT)
    if d.startswith("sub-")
])

# ==========================
# HELPERS
# ==========================

def parse_marker(marker):

    info = {}

    parts = marker.split(";")

    for part in parts:

        if "=" not in part:
            continue

        k, v = part.split("=", 1)

        info[k.strip()] = v.strip()

    return info

# ==========================
# EXPORT
# ==========================

for sub in SUBJECTS:

    print(f"\n=== Exporting {sub} ===")

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
    y_freq_all = []
    y_target_all = []
    groups = []

    metadata_rows = []

    for ses in sessions:

        print("Reading", ses)

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

        raw.pick(SSVEP_CHANNELS)

        raw.load_data()

        raw.filter(
            BANDPASS[0],
            BANDPASS[1],
            fir_design="firwin",
            verbose=False
        )

        df_ann = raw.annotations.to_data_frame()

        df_ann = df_ann.rename(
            columns={"description": "trial_type"}
        )

        first_onset = df_ann["onset"].iloc[0]

        start_events = []

        for _, row in df_ann.iterrows():

            marker = str(row["trial_type"])

            if "event=start" not in marker:
                continue

            info = parse_marker(marker)

            if (
                "trial" not in info
                or
                "freq" not in info
                or
                "target" not in info
            ):
                continue

            start_events.append(
                (row, info)
            )

        for row, info in start_events:

            onset = (
                row["onset"] - first_onset
            ).total_seconds()

            start_samp = int(
                onset * SFREQ
            )

            stop_samp = (
                start_samp
                + TRIAL_SAMPLES
            )

            epoch = raw.get_data(
                start=start_samp,
                stop=stop_samp
            )

            if epoch.shape[1] != TRIAL_SAMPLES:

                print(
                    f"Skipping incomplete trial "
                    f"{info.get('trial','?')}"
                )

                continue

            # --------------------------------
            # Normalize per channel
            # --------------------------------

            epoch = (
                epoch
                - epoch.mean(
                    axis=1,
                    keepdims=True
                )
            ) / (
                epoch.std(
                    axis=1,
                    keepdims=True
                )
                + 1e-8
            )

            freq = int(info["freq"])

            target = info["target"].lower()

            # -------------------------------
            # Frequency label
            # -------------------------------

            if freq == 12:

                y_freq = 0

            elif freq == 15:

                y_freq = 1

            else:

                print(
                    f"Unknown frequency {freq}"
                )

                continue

            # -------------------------------
            # Target label
            # -------------------------------

            if target == "bottom":

                y_target = 0

            elif target == "top":

                y_target = 1

            else:

                print(
                    f"Unknown target {target}"
                )

                continue

            X_all.append(epoch)

            y_freq_all.append(y_freq)

            y_target_all.append(y_target)

            groups.append(ses)

            metadata_rows.append({
                "subject": sub,
                "session": ses,
                "trial": info["trial"],
                "freq": freq,
                "target": target,
                "duration_s": TRIAL_DURATION_SEC
            })

    X_all = np.array(X_all)

    y_freq_all = np.array(
        y_freq_all
    )

    y_target_all = np.array(
        y_target_all
    )

    groups = np.array(groups)

    print(
        f"X shape: {X_all.shape}"
    )

    print(
        f"12Hz: {(y_freq_all==0).sum()} "
        f"15Hz: {(y_freq_all==1).sum()}"
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
        os.path.join(
            subject_out,
            "X.npy"
        ),
        X_all
    )

    np.save(
        os.path.join(
            subject_out,
            "y_freq.npy"
        ),
        y_freq_all
    )

    np.save(
        os.path.join(
            subject_out,
            "y_target.npy"
        ),
        y_target_all
    )

    np.save(
        os.path.join(
            subject_out,
            "groups.npy"
        ),
        groups
    )

    pd.DataFrame(
        metadata_rows
    ).to_csv(
        os.path.join(
            subject_out,
            "metadata.csv"
        ),
        index=False
    )

    print(
        f"Saved {subject_out}"
    )

print("\nDONE")