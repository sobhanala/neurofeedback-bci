#this is to see if the prev 100 and 101 model can work on the new 200 and 201
import json
import pickle
import numpy as np

# =====================================
# CONFIG
# =====================================

MODEL_SUBJECT = "P101"
TEST_SUBJECT = "P201"

# Examples:
#MODEL_SUBJECT = "P100"
#TEST_SUBJECT = "P200"

# =====================================
# LOAD MODEL
# =====================================

MODEL_DIR = f"final_models/{MODEL_SUBJECT}"

with open(f"{MODEL_DIR}/csp_models.pkl", "rb") as f:
    csp_models = pickle.load(f)

with open(f"{MODEL_DIR}/selector.pkl", "rb") as f:
    selector = pickle.load(f)

with open(f"{MODEL_DIR}/svm.pkl", "rb") as f:
    svm = pickle.load(f)

with open(f"{MODEL_DIR}/model_info.json", "r") as f:
    model_info = json.load(f)

print("\nLoaded model:")
print("Model Subject:", MODEL_SUBJECT)

# =====================================
# LOAD TEST DATA
# =====================================

X = np.load(
    f"dataset/subject_{TEST_SUBJECT}/X.npy"
)

y = np.load(
    f"dataset/subject_{TEST_SUBJECT}/y.npy"
)

print("\nLoaded test data:")
print("Test Subject:", TEST_SUBJECT)
print("X shape:", X.shape)

# =====================================
# FEATURE EXTRACTION
# =====================================
from scipy.signal import butter, filtfilt

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

bands = model_info["bands"]

feature_blocks = []

for (low, high), csp in zip(
    bands,
    csp_models
):

    X_band = np.array([
        bandpass_epoch(
            epoch,
            low,
            high,
            model_info["sfreq"]
        )
        for epoch in X
    ])

    X_csp = csp.transform(
        X_band
    )

    feature_blocks.append(
        X_csp
    )

X_features = np.concatenate(
    feature_blocks,
    axis=1
)

X_features = selector.transform(
    X_features
)

# =====================================
# PREDICT
# =====================================

y_pred = svm.predict(
    X_features
)

accuracy = np.mean(
    y_pred == y
)

# =====================================
# RESULTS
# =====================================

print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)

print("Model :", MODEL_SUBJECT)
print("Test  :", TEST_SUBJECT)

print(
    f"Accuracy: {accuracy:.3f}"
)

left_acc = np.mean(
    y_pred[y == 0] == 0
)

right_acc = np.mean(
    y_pred[y == 1] == 1
)

print(
    f"Left accuracy : {left_acc:.3f}"
)

print(
    f"Right accuracy: {right_acc:.3f}"
)