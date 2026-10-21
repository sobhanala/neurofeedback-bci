# batch_xdf_to_bids.py
import os
import numpy as np
import pyxdf
import mne
from mne_bids import BIDSPath, write_raw_bids
import pandas as pd

# ================= CONFIG ==================
DATA_ROOT = r"dataaaaa"         # folder containing all sub-PXXX folders
BIDS_ROOT = r"MI_BIDS"          # output BIDS folder
MONTAGE = None                   # e.g., mne.channels.make_standard_montage("standard_1020")
SFREQ_OVERRIDE = 256             # optional
SUMMARY_CSV = os.path.join(BIDS_ROOT, "dataset_summary.csv")
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
            if file.endswith(".xdf"):
                xdf_path = os.path.join(ses_path, file)
                print(f"Processing {xdf_path} ...")

                # Load XDF
                streams, header = pyxdf.load_xdf(xdf_path, dejitter_timestamps=True)

                eeg = None
                mrk = None
                for s in streams:
                    t = s["info"]["type"][0].upper()
                    if t == "EEG" and eeg is None:
                        eeg = s
                    if t in ("MARKERS", "MARKER", "EVENTS") and mrk is None:
                        mrk = s
                if eeg is None:
                    print("No EEG stream found, skipping.")
                    continue

                # Sampling rate
                sfreq = float(SFREQ_OVERRIDE) if SFREQ_OVERRIDE else float(eeg["info"]["nominal_srate"][0])

                # Channels
                try:
                    ch_names = [c["label"][0] for c in eeg["info"]["desc"][0]["channels"][0]["channel"]]
                except Exception:
                    ch_names = [f"EEG{i+1}" for i in range(eeg["time_series"].shape[1])]
                ch_types = ["eeg"] * len(ch_names)

                # Build Raw
                data = np.asarray(eeg["time_series"], dtype=float).T
                info = mne.create_info(ch_names, sfreq, ch_types)
                raw = mne.io.RawArray(data, info)
                if MONTAGE is not None:
                    raw.set_montage(MONTAGE)
                raw.set_meas_date(None)

                # Parse markers and normalize names
                EVENT_ID = {}
                descs = []
                if mrk is not None:
                    onsets_lsl = np.asarray(mrk["time_stamps"], dtype=float)
                    for v in mrk["time_series"]:
                        val = v[0] if isinstance(v, (list, tuple, np.ndarray)) else v
                        val = str(val)
                        # Normalize marker names: map "imagine_on" → "imagery_on"
                        val = val.replace("imagine_on", "imagery_on").replace("imagine_end", "imagery_end")
                        descs.append(val)
                    raw.set_annotations(mne.Annotations(
                        onset=(onsets_lsl - eeg["time_stamps"][0]),
                        duration=np.zeros_like(onsets_lsl),
                        description=descs
                    ))
                    unique_descs = sorted(set(descs))
                    EVENT_ID = {desc: i + 1 for i, desc in enumerate(unique_descs)}

                # Build BIDS path
                bids_path = BIDSPath(subject=sub.replace("sub-", ""),
                                     session=ses.replace("ses-", ""),
                                     task="mi",
                                     root=BIDS_ROOT,
                                     datatype="eeg",
                                     suffix="eeg")
                # Write BIDS
                write_raw_bids(raw, bids_path=bids_path, overwrite=True,
                               format="BrainVision", allow_preload=True, event_id=EVENT_ID)
                print(f"BIDS written for {sub} {ses}")

                # Count trials for summary
                # Corrected counting for guided trials
                guided_left = sum(1 for d in descs if "cond=left" in d)
                guided_right = sum(1 for d in descs if "cond=right" in d)
                
                # Corrected counting for free-choice trials
                # Count only once per free-choice trial using imagery_start markers
                free_choice = sum(1 for d in descs if "free_trial=" in d and "imagery_start" in d)
                
                summary_rows.append({
                    "subject": sub,
                    "session": ses,
                    "duration_s": raw.n_times / sfreq,
                    "guided_left": guided_left,
                    "guided_right": guided_right,
                    "free_choice": free_choice
                })

# Save summary CSV
os.makedirs(BIDS_ROOT, exist_ok=True)
df = pd.DataFrame(summary_rows)
df.to_csv(SUMMARY_CSV, index=False)
print(f"Dataset summary saved to {SUMMARY_CSV}")
print("All XDF files processed and BIDS dataset is ready!")