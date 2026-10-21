# realtime_mi_ssvep_gate.py

import json
import pickle
import time

import numpy as np

from scipy.signal import welch
from scipy.signal import butter
from scipy.signal import filtfilt

from pylsl import StreamInlet
from pylsl import resolve_byprop

# ==================================================
# CONFIG
# ==================================================

FS = 256

WINDOW_SAMPLES = 512      # 2 sec
UPDATE_SAMPLES = 64       # 250 ms

MODEL_DIR = "gate_model"

# ==================================================
# LOAD MODEL
# ==================================================

with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(f"{MODEL_DIR}/lda.pkl", "rb") as f:
    clf = pickle.load(f)

with open(f"{MODEL_DIR}/model_info.json", "r") as f:
    model_info = json.load(f)

print("\n==============================")
print("MI vs SSVEP GATE")
print("==============================")

print("Loaded model")

print("Classes:")
print(clf.classes_)

print("Sampling Rate:", FS)

# ==================================================
# FILTER
# ==================================================

def bandpass_epoch(epoch):

    nyq = FS / 2

    b, a = butter(
        4,
        [8 / nyq, 45 / nyq],
        btype="band"
    )

    return filtfilt(
        b,
        a,
        epoch,
        axis=1
    )

# ==================================================
# FEATURE EXTRACTION
# ==================================================

def extract_psd_features(epoch):

    # --------------------------------
    # Match offline training as much
    # as possible
    # --------------------------------

    epoch = bandpass_epoch(epoch)

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
    feats = []

    for ch in range(epoch.shape[0]):

        freqs, psd = welch(
            epoch[ch],
            fs=FS,
            nperseg=256
        )

        bands = [
            (4, 8),
            (8, 13),
            (13, 30),
            (30, 45)
        ]

        for low, high in bands:

            idx = (
                (freqs >= low)
                &
                (freqs < high)
            )

            feats.append(
                np.mean(psd[idx])
            )

    return np.array(feats)

# ==================================================
# CONNECT TO EEG
# ==================================================

print("\nWaiting for EmotivEEG stream...")

streams = []

while len(streams) == 0:

    streams = resolve_byprop(
        "name",
        "EmotivEEG",
        timeout=2
    )

    if len(streams) == 0:

        print("Still waiting...")
        time.sleep(1)

print("Connected.")

inlet = StreamInlet(
    streams[0]
)

# ==================================================
# BUFFER
# ==================================================

buffer = np.empty(
    (14, 0),
    dtype=np.float32
)

samples_since_prediction = 0

# ==================================================
# REALTIME LOOP
# ==================================================

print("\nRealtime gate started.")
print("Press CTRL+C to stop.\n")

print("\nTEST CONDITIONS:")
print("1) Rest")
print("2) Strong MI")
print("3) Look at 15 Hz flicker\n")

try:

    while True:

        sample, timestamp = inlet.pull_sample()

        sample = np.array(
            sample,
            dtype=np.float32
        ).reshape(14, 1)

        buffer = np.concatenate(
            [buffer, sample],
            axis=1
        )

        if buffer.shape[1] > WINDOW_SAMPLES:

            buffer = buffer[:, -WINDOW_SAMPLES:]

        if buffer.shape[1] < WINDOW_SAMPLES:

            continue

        samples_since_prediction += 1

        if samples_since_prediction < UPDATE_SAMPLES:

            continue

        samples_since_prediction = 0

        # --------------------------------
        # Features
        # --------------------------------

        feat = extract_psd_features(
            buffer
        )
        print(
            "RAW FEATURE RANGE:",
            feat.min(),
            feat.max(),
            feat.mean()
        )
        feat = scaler.transform(
            [feat]
        )

        # --------------------------------
        # Prediction
        # --------------------------------

        probs = clf.predict_proba(
            feat
        )[0]

        pred = clf.predict(
            feat
        )[0]

        score = clf.decision_function(
            feat
        )[0]

        if pred == 0:

            state = "MI"

        else:

            state = "SSVEP"

        # --------------------------------
        # Print diagnostics
        # --------------------------------

        print(f"SCORE={score:+8.3f}")

except KeyboardInterrupt:

    print("\nStopped.")