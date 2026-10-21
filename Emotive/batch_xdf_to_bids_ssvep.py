# batch_xdf_to_bids_ssvep.py

import os
import numpy as np
import pyxdf
import mne
from mne_bids import BIDSPath, write_raw_bids
import pandas as pd

# ================= CONFIG ==================
DATA_ROOT = r"ssvep_data"          # folder containing sub-PXXX folders
BIDS_ROOT = r"SSVEP_BIDS"          # output BIDS folder
MONTAGE = None                     # optional montage
SFREQ_OVERRIDE = 256               # optional
SUMMARY_CSV = os.path.join(BIDS_ROOT, "ssvep_dataset_summary.csv")
# ==========================================

summary_rows = []

for sub in sorted(os.listdir(DATA_ROOT)):
    sub_path = os.path.join(DATA_ROOT, sub)
    if not os.path.isdir(sub_path):
        continue

    for ses in sorted(os.listdir(sub_path)):
        ses_path = os.path.join(sub_path, ses, "eeg")

        if not os.path.isdir(ses_path):
            continue

        for file in os.listdir(ses_path):

            if not file.endswith(".xdf"):
                continue

            xdf_path = os.path.join(ses_path, file)

            print("=" * 80)
            print(f"Processing: {xdf_path}")

            # -------------------------------------------------
            # Load XDF
            # -------------------------------------------------
            streams, header = pyxdf.load_xdf(
                xdf_path,
                dejitter_timestamps=True
            )

            eeg = None
            mrk = None

            for s in streams:

                stream_type = s["info"]["type"][0].upper()

                if stream_type == "EEG" and eeg is None:
                    eeg = s

                if stream_type in ("MARKERS", "MARKER", "EVENTS") and mrk is None:
                    mrk = s

            if eeg is None:
                print("No EEG stream found. Skipping.")
                continue

            # -------------------------------------------------
            # Sampling rate
            # -------------------------------------------------
            sfreq = (
                float(SFREQ_OVERRIDE)
                if SFREQ_OVERRIDE
                else float(eeg["info"]["nominal_srate"][0])
            )

            # -------------------------------------------------
            # Channel names
            # -------------------------------------------------
            try:
                ch_names = [
                    c["label"][0]
                    for c in eeg["info"]["desc"][0]["channels"][0]["channel"]
                ]
            except Exception:
                ch_names = [
                    f"EEG{i+1}"
                    for i in range(eeg["time_series"].shape[1])
                ]

            ch_types = ["eeg"] * len(ch_names)

            # -------------------------------------------------
            # Build Raw object
            # -------------------------------------------------
            data = np.asarray(
                eeg["time_series"],
                dtype=float
            ).T

            info = mne.create_info(
                ch_names,
                sfreq,
                ch_types
            )

            raw = mne.io.RawArray(data, info)

            if MONTAGE is not None:
                raw.set_montage(MONTAGE)

            raw.set_meas_date(None)

            # -------------------------------------------------
            # Markers -> MNE annotations
            # -------------------------------------------------
            EVENT_ID = {}
            descs = []

            if mrk is not None:

                onsets_lsl = np.asarray(
                    mrk["time_stamps"],
                    dtype=float
                )

                for value in mrk["time_series"]:

                    marker = (
                        value[0]
                        if isinstance(value, (list, tuple, np.ndarray))
                        else value
                    )

                    marker = str(marker).strip()

                    descs.append(marker)

                raw.set_annotations(
                    mne.Annotations(
                        onset=(onsets_lsl - eeg["time_stamps"][0]),
                        duration=np.zeros_like(onsets_lsl),
                        description=descs
                    )
                )

                unique_descs = sorted(set(descs))

                EVENT_ID = {
                    desc: idx + 1
                    for idx, desc in enumerate(unique_descs)
                }

                print("\nDetected markers:")
                for marker in unique_descs:
                    print(f"   {marker}")

            else:
                print("No marker stream found.")

            # -------------------------------------------------
            # Build BIDS path
            # -------------------------------------------------
            bids_path = BIDSPath(
                subject=sub.replace("sub-", ""),
                session=ses.replace("ses-", ""),
                task="ssvep",
                root=BIDS_ROOT,
                datatype="eeg",
                suffix="eeg"
            )

            # -------------------------------------------------
            # Write BIDS
            # -------------------------------------------------
            write_raw_bids(
                raw,
                bids_path=bids_path,
                overwrite=True,
                format="BrainVision",
                allow_preload=True,
                event_id=EVENT_ID
            )

            print(f"BIDS written for {sub} {ses}")

            summary_rows.append({
                "subject": sub,
                "session": ses,
                "duration_s": raw.n_times / sfreq,
                "n_channels": len(ch_names),
                "n_markers": len(descs),
                "unique_markers": len(set(descs))
            })

# -------------------------------------------------
# Save summary
# -------------------------------------------------
os.makedirs(BIDS_ROOT, exist_ok=True)

df = pd.DataFrame(summary_rows)
df.to_csv(SUMMARY_CSV, index=False)

print("\nDataset summary saved:")
print(SUMMARY_CSV)

print("\nAll SSVEP XDF files processed successfully.")