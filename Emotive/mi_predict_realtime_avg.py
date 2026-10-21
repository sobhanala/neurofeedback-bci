import json
import pickle
import time

import numpy as np

from scipy.signal import butter, filtfilt

from pylsl import StreamInlet
from pylsl import resolve_byprop

# =====================================
# CONFIG
# =====================================
REST_TIME = 5.0
TRIAL_TIME = 5.0

UPDATE_SAMPLES = 64

#MODEL_SUBJECT = "P100"
#MODEL_SUBJECT = "P101"
#MODEL_SUBJECT = "P200"
MODEL_SUBJECT = "P201"

MODEL_DIR = f"final_models/{MODEL_SUBJECT}"

WINDOW_SAMPLES = 512
UPDATE_SAMPLES = 64

BANDS = [
    (8, 12),
    (12, 16),
    (16, 20),
    (20, 24),
    (24, 30)
]

# =====================================
# LOAD MODELS
# =====================================

with open(f"{MODEL_DIR}/model_info.json", "r") as f:
    model_info = json.load(f)

with open(f"{MODEL_DIR}/csp_models.pkl", "rb") as f:
    csp_models = pickle.load(f)

with open(f"{MODEL_DIR}/selector.pkl", "rb") as f:
    selector = pickle.load(f)

with open(f"{MODEL_DIR}/svm.pkl", "rb") as f:
    svm = pickle.load(f)

SFREQ = model_info["sfreq"]

print("\nLoaded model:")
print("Subject:", MODEL_SUBJECT)
print("Sampling Rate:", SFREQ)
print("Channels:", len(model_info["channels"]))
print("Bands:", model_info["bands"])

# =====================================
# BANDPASS
# =====================================

def bandpass_epoch(epoch, low, high, fs):

    nyq = fs / 2

    b, a = butter(
        4,
        [low / nyq, high / nyq],
        btype="band"
    )

    return filtfilt(
        b,
        a,
        epoch,
        axis=1
    )

# =====================================
# FEATURE EXTRACTION
# =====================================

def extract_features(epoch):

    # epoch shape:
    # (14,512)

    epoch = (
        epoch
        - epoch.mean(axis=1, keepdims=True)
    ) / (
        epoch.std(axis=1, keepdims=True)
        + 1e-8
    )

    feature_blocks = []

    for (low, high), csp in zip(
        BANDS,
        csp_models
    ):

        epoch_band = bandpass_epoch(
            epoch,
            low,
            high,
            SFREQ
        )

        epoch_band = epoch_band[np.newaxis, :, :]

        csp_features = csp.transform(
            epoch_band
        )

        feature_blocks.append(
            csp_features
        )

    X = np.concatenate(
        feature_blocks,
        axis=1
    )

    X = selector.transform(X)

    return X

# =====================================
# CONNECT TO EEG STREAM
# =====================================

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

# =====================================
# BUFFER
# =====================================

buffer = np.empty(
    (14, 0),
    dtype=np.float32
)

samples_since_prediction = 0

# =====================================
# REALTIME LOOP
# =====================================


print("\nRealtime prediction started.")
print("Press CTRL+C to stop.\n")

print("\n==============================")
print("MOTOR IMAGERY TRIAL")
print("==============================")

trial_predictions = []
trial_probabilities = []
try:

    while True:

        print("\nREST")
        time.sleep(REST_TIME)

        print("\nSTART IMAGERY NOW")
        print("Think LEFT or RIGHT")

        trial_predictions = []
        trial_probabilities = []

        trial_start = time.time()

        while time.time() - trial_start < TRIAL_TIME:

            sample, timestamp = inlet.pull_sample()

            sample = np.array(
                sample,
                dtype=np.float32
            ).reshape(14,1)

            buffer = np.concatenate(
                [buffer, sample],
                axis=1
            )

            if buffer.shape[1] > WINDOW_SAMPLES:

                buffer = buffer[:, -WINDOW_SAMPLES:]

            samples_since_prediction += 1

            if (
                buffer.shape[1] == WINDOW_SAMPLES
                and
                samples_since_prediction >= UPDATE_SAMPLES
            ):

                samples_since_prediction = 0

                X = extract_features(buffer)

                pred = svm.predict(X)[0]

                probs = svm.predict_proba(X)[0]

                trial_predictions.append(pred)

                trial_probabilities.append(probs)
        if len(trial_predictions) == 0:

            print("No predictions made.")
            continue
        
        trial_predictions = np.array(
            trial_predictions
        )

        trial_probabilities = np.array(
            trial_probabilities
        )

        left_votes = np.sum(
            trial_predictions == 0
        )

        right_votes = np.sum(
            trial_predictions == 1
        )

        avg_probs = np.mean(
            trial_probabilities,
            axis=0
        )

        if left_votes > right_votes:
        
            final_label = "LEFT"

            confidence = avg_probs[0]

        else:
        
            final_label = "RIGHT"

            confidence = avg_probs[1]

        print("\n==============================")
        print("FINAL DECISION")
        print("==============================")

        print(f"LEFT votes : {left_votes}")
        print(f"RIGHT votes: {right_votes}")
        
        print(f"P(LEFT)  = {avg_probs[0]:.3f}")
        print(f"P(RIGHT) = {avg_probs[1]:.3f}")

        print(
            f"Prediction: {final_label}"
        )

        print(
            f"Confidence: {confidence:.3f}"
        )
except KeyboardInterrupt:

    print("\nStopped.")

    