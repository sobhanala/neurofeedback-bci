# xdf_to_bids.py  — XDF -> BIDS (BrainVision) minimal converter
import os
import numpy as np
import mne
import pyxdf
from mne_bids import BIDSPath, write_raw_bids

# --------- EDIT THESE -----------
XDF_PATH = r"C:\Users\M.Kolahdoozan\Desktop\amin\MIBCI project\MIBCI\Emotiv Data Acquisition\summary\dataaaaa\sub-P100\ses-S001\eeg\sub-P100_ses-S001_task-Default_run-001_eeg.xdf"
SUB = "100"
SES = "01"
TASK = "mi"
OUT_BIDS = "bids_out"
MONTAGE = None             # e.g., mne.channels.make_standard_montage("standard_1020")
# --------------------------------

print(f"Loading {XDF_PATH} ...")
streams, header = pyxdf.load_xdf(XDF_PATH, dejitter_timestamps=True)

###temporary code to see iof free trail markers are there 
for s in streams:
    if s["info"]["type"][0].upper() == "MARKERS":
        print("\n=== ALL UNIQUE MARKERS ===")

        unique_markers = set()

        for m in s["time_series"]:
            value = m[0] if isinstance(m, (list, tuple, np.ndarray)) else m
            unique_markers.add(str(value))

        for marker in sorted(unique_markers):
            print(marker)

# Find streams by LSL 'type'
def _type(s): return s["info"]["type"][0].upper()
try:
    eeg = next(s for s in streams if _type(s) == "EEG")
    mrk = next(s for s in streams if _type(s) == "MARKERS")
except StopIteration:
    raise RuntimeError("Could not find both EEG and MARKERS streams in the XDF.")

# EEG data
sfreq = float(eeg["info"]["nominal_srate"][0])
ch_names = [c["label"][0] for c in eeg["info"]["desc"][0]["channels"][0]["channel"]]
ch_types = ["eeg"] * len(ch_names)

data = np.asarray(eeg["time_series"], dtype=float).T  # (n_channels, n_samples)
times = np.asarray(eeg["time_stamps"], dtype=float)

# Build Raw
info = mne.create_info(ch_names, sfreq, ch_types)
raw = mne.io.RawArray(data, info)
if MONTAGE is not None:
    raw.set_montage(MONTAGE)
raw.set_meas_date(None)  # avoid tz warnings, use relative time

# Parse Markers like "trial=5;cond=left;event=cue_on" OR "cue_on/left"
def parse_marker(payload: str):
    s = str(payload)
    if ";" in s and "=" in s:
        parts = dict(p.split("=", 1) for p in s.split(";") if "=" in p)
        event = parts.get("event", "event")
        cond = parts.get("cond", "NA")
        return f"{event}/{cond}"
    return s  # passthrough e.g., "cue_on/left"

onsets_lsl = np.asarray(mrk["time_stamps"], dtype=float)
descs = [parse_marker(v[0] if isinstance(v, (list, tuple)) else v) for v in mrk["time_series"]]

# Convert absolute LSL times to sample indices relative to EEG stream start
t0 = times[0]
onset_samples = (sfreq * (onsets_lsl - t0)).astype(int)
onset_secs = onset_samples / sfreq

raw.set_annotations(mne.Annotations(onset=onset_secs,
                                    duration=np.zeros_like(onset_secs, dtype=float),
                                    description=descs))


# Build a stable event_id mapping from the annotations already in `raw`
unique_descs = sorted(set(raw.annotations.description))
EVENT_ID = {desc: i + 1 for i, desc in enumerate(unique_descs)}

# Write BIDS (BrainVision .vhdr/.eeg/.vmrk)
bids_path = BIDSPath(subject=SUB, session=SES, task=TASK, root=OUT_BIDS, datatype="eeg", suffix="eeg")
write_raw_bids(raw, bids_path=bids_path, overwrite=True, format="BrainVision", allow_preload=True, event_id=EVENT_ID)
print(f"BIDS dataset written to: {OUT_BIDS}")
