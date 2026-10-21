import numpy as np
import pickle

from scipy.signal import welch

FS = 256

def extract_psd_features(epoch):

    feats = []

    for ch in range(epoch.shape[0]):

        freqs, psd = welch(
            epoch[ch],
            fs=FS,
            nperseg=256
        )

        bands = [
            (4,8),
            (8,13),
            (13,30),
            (30,45)
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

X = np.load(
    "dataset/subject_P200/X.npy"
)

epoch = X[0]

feat = extract_psd_features(epoch)

print("\nTraining sample")
print("Min:", feat.min())
print("Max:", feat.max())
print("Mean:", feat.mean())

with open(
    "gate_model/scaler.pkl",
    "rb"
) as f:

    scaler = pickle.load(f)

feat_scaled = scaler.transform(
    [feat]
)

print("\nAfter scaler")
print("Min:", feat_scaled.min())
print("Max:", feat_scaled.max())
print("Mean:", feat_scaled.mean())