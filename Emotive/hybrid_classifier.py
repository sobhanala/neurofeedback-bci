# ==========================================================
# HYBRID CLASSIFIER
#
# Purpose:
#   MI + SSVEP Gate classifier service
#
# Architecture:
#
# initialize()
# run_trial()
# shutdown()
#
# Fusion owns timing.
# Classifier owns EEG + prediction.
# ==========================================================

import os
import csv
import json
import pickle
import time

from datetime import datetime

import numpy as np

from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import welch

from pylsl import StreamInlet
from pylsl import resolve_byprop
from presenter_override import PresenterOverride

class HybridClassifier:

    # ======================================================
    # CONFIG
    # ======================================================

    MI_SUBJECT_MODEL = "P201"
    # MI_SUBJECT_MODEL = "P200"

    FS = 256

    WINDOW_SAMPLES = 512
    UPDATE_SAMPLES = 64

    TRIAL_TIME = 5.0

    ENABLE_LOGGING = True

    ENABLE_JSON_OUTPUT = True

    LOG_FILE = "prediction_log.csv"

    JSON_FILE = "latest_prediction.json"

    GATE_MODEL_DIR = "gate_model"
    
    LSL_STREAM_NAME = "EmotivEEG"
    # ======================================================
    # INIT
    # ======================================================

    def __init__(self):

        self.initialized = False

        self.inlet = None

        # -------------------------
        # MI models
        # -------------------------

        self.csp_models = None

        self.selector = None

        self.svm = None

        self.mi_model_info = None

        # -------------------------
        # Gate models
        # -------------------------

        self.scaler = None

        self.lda = None

        self.gate_model_info = None

        self.presenter = PresenterOverride()

    # ======================================================
    # INITIALIZE
    # ======================================================

    def initialize(self):

        if self.initialized:

            print(
                "Already initialized."
            )

            return

        print("\n================================")
        print("INITIALIZING HYBRID CLASSIFIER")
        print("================================")

        self._load_mi_model()

        self._load_gate_model()

        self._connect_lsl()

        self.initialized = True

        print("\nInitialization complete.")

    # ======================================================
    # LOAD MI MODEL
    # ======================================================

    def _load_mi_model(self):

        print(
            "\nLoading MI model..."
        )

        model_dir = (
            f"final_models/"
            f"{self.MI_SUBJECT_MODEL}"
        )

        with open(
            f"{model_dir}/model_info.json",
            "r"
        ) as f:

            self.mi_model_info = json.load(f)

        with open(
            f"{model_dir}/csp_models.pkl",
            "rb"
        ) as f:

            self.csp_models = pickle.load(f)

        with open(
            f"{model_dir}/selector.pkl",
            "rb"
        ) as f:

            self.selector = pickle.load(f)

        with open(
            f"{model_dir}/svm.pkl",
            "rb"
        ) as f:

            self.svm = pickle.load(f)

        print(
            f"Loaded MI model "
            f"{self.MI_SUBJECT_MODEL}"
        )

    # ======================================================
    # LOAD GATE MODEL
    # ======================================================

    def _load_gate_model(self):

        print(
            "\nLoading Gate model..."
        )

        with open(
            f"{self.GATE_MODEL_DIR}/scaler.pkl",
            "rb"
        ) as f:

            self.scaler = pickle.load(f)

        with open(
            f"{self.GATE_MODEL_DIR}/lda.pkl",
            "rb"
        ) as f:

            self.lda = pickle.load(f)

        with open(
            f"{self.GATE_MODEL_DIR}/model_info.json",
            "r"
        ) as f:

            self.gate_model_info = json.load(f)

        print(
            "Loaded MI/SSVEP gate."
        )

    # ======================================================
    # CONNECT TO EEG
    # ======================================================

    def _connect_lsl(self):

        print(
            "\nWaiting for EmotivEEG..."
        )

        streams = []

        while len(streams) == 0:

            streams = resolve_byprop(
                "name",
                self.LSL_STREAM_NAME,
                timeout=2
            )

            if len(streams) == 0:

                print(
                    "Still waiting..."
                )

                time.sleep(1)

        self.inlet = StreamInlet(
            streams[0]
        )

        print(
            "Connected to EmotivEEG."
        )

    # ======================================================
    # BANDPASS
    # ======================================================

    def bandpass_epoch(
        self,
        epoch,
        low,
        high,
        fs
    ):

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

    # ======================================================
    # GATE FILTER
    # ======================================================

    def gate_bandpass(
        self,
        epoch
    ):

        nyq = self.FS / 2

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

    # ======================================================
    # MI FEATURES
    # ======================================================

    def extract_mi_features(
        self,
        epoch
    ):

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

        feature_blocks = []

        bands = [
            (8,12),
            (12,16),
            (16,20),
            (20,24),
            (24,30)
        ]

        for (
            (low, high),
            csp
        ) in zip(
            bands,
            self.csp_models
        ):

            epoch_band = (
                self.bandpass_epoch(
                    epoch,
                    low,
                    high,
                    self.FS
                )
            )

            epoch_band = (
                epoch_band[
                    np.newaxis,
                    :,
                    :
                ]
            )

            csp_features = (
                csp.transform(
                    epoch_band
                )
            )

            feature_blocks.append(
                csp_features
            )

        X = np.concatenate(
            feature_blocks,
            axis=1
        )

        X = self.selector.transform(
            X
        )

        return X

    # ======================================================
    # GATE FEATURES
    # ======================================================

    def extract_gate_features(
        self,
        epoch
    ):

        epoch = self.gate_bandpass(
            epoch
        )

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

        bands = [
            (4, 8),
            (8, 13),
            (13, 30),
            (30, 45)
        ]

        for ch in range(
            epoch.shape[0]
        ):

            freqs, psd = welch(
                epoch[ch],
                fs=self.FS,
                nperseg=256
            )

            for low, high in bands:

                idx = (
                    (freqs >= low)
                    &
                    (freqs < high)
                )

                feats.append(
                    np.mean(
                        psd[idx]
                    )
                )

        return np.array(feats)

    # ======================================================
    # RUN TRIAL
    # ======================================================

    def _run_trial_core(self):

        if not self.initialized:

            raise RuntimeError(
                "Call initialize() first."
            )

        print("\n================================")
        print("STARTING NEW TRIAL")
        print("================================")

        # ----------------------------------
        # Fresh buffer every trial
        # ----------------------------------

        buffer = np.empty(
            (14, 0),
            dtype=np.float32
        )

        samples_since_prediction = 0

        # ----------------------------------
        # MI storage
        # ----------------------------------

        mi_predictions = []

        mi_probabilities = []

        # ----------------------------------
        # Gate storage
        # ----------------------------------

        gate_predictions = []

        gate_probabilities = []

        # ----------------------------------
        # Timing
        # ----------------------------------

        trial_start = time.time()

        # ----------------------------------
        # Collect data
        # ----------------------------------

        while (
            time.time()
            - trial_start
        ) < self.TRIAL_TIME:

            sample, timestamp = (
                self.inlet.pull_sample()
            )

            sample = np.array(
                sample,
                dtype=np.float32
            ).reshape(14, 1)

            buffer = np.concatenate(
                [
                    buffer,
                    sample
                ],
                axis=1
            )

            if (
                buffer.shape[1]
                > self.WINDOW_SAMPLES
            ):

                buffer = buffer[
                    :,
                    -self.WINDOW_SAMPLES:
                ]

            if (
                buffer.shape[1]
                < self.WINDOW_SAMPLES
            ):

                continue

            samples_since_prediction += 1

            if (
                samples_since_prediction
                < self.UPDATE_SAMPLES
            ):

                continue

            samples_since_prediction = 0

            current_window = (
                buffer.copy()
            )

            # ==================================
            # MI PREDICTION
            # ==================================

            mi_X = (
                self.extract_mi_features(
                    current_window
                )
            )

            mi_pred = (
                self.svm.predict(
                    mi_X
                )[0]
            )

            mi_probs = (
                self.svm.predict_proba(
                    mi_X
                )[0]
            )

            mi_predictions.append(
                mi_pred
            )

            mi_probabilities.append(
                mi_probs
            )

            # ==================================
            # GATE PREDICTION
            # ==================================

            gate_feat = (
                self.extract_gate_features(
                    current_window
                )
            )

            gate_feat = (
                self.scaler.transform(
                    [gate_feat]
                )
            )

            gate_pred = (
                self.lda.predict(
                    gate_feat
                )[0]
            )

            gate_probs = (
                self.lda.predict_proba(
                    gate_feat
                )[0]
            )

            gate_predictions.append(
                gate_pred
            )

            gate_probabilities.append(
                gate_probs
            )

        # ==================================
        # SAFETY CHECK
        # ==================================

        if len(mi_predictions) == 0:

            raise RuntimeError(
                "No predictions generated."
            )

        # ==================================
        # MAJORITY VOTE - MI
        # ==================================

        mi_predictions = np.array(
            mi_predictions
        )

        mi_probabilities = np.array(
            mi_probabilities
        )

        left_votes = np.sum(
            mi_predictions == 0
        )

        right_votes = np.sum(
            mi_predictions == 1
        )

        avg_mi_probs = np.mean(
            mi_probabilities,
            axis=0
        )

        if left_votes > right_votes:

            mi_label = "LEFT"

            mi_confidence = (
                avg_mi_probs[0]
            )

        else:

            mi_label = "RIGHT"

            mi_confidence = (
                avg_mi_probs[1]
            )

        # ==================================
        # MAJORITY VOTE - GATE
        # ==================================

        gate_predictions = np.array(
            gate_predictions
        )

        gate_probabilities = np.array(
            gate_probabilities
        )

        mi_gate_votes = np.sum(
            gate_predictions == 0
        )

        ssvep_gate_votes = np.sum(
            gate_predictions == 1
        )

        avg_gate_probs = np.mean(
            gate_probabilities,
            axis=0
        )

        if (
            mi_gate_votes
            >
            ssvep_gate_votes
        ):

            gate_label = "MI"

            gate_confidence = (
                avg_gate_probs[0]
            )

        else:

            gate_label = "SSVEP"

            gate_confidence = (
                avg_gate_probs[1]
            )

        # ==================================
        # FUSION LOGIC
        # ==================================

        if gate_label == "SSVEP":

            final_prediction = "UP"

        else:

            final_prediction = (
                mi_label
            )
        # ==================================
        override = self.presenter.consume_override()

        if override is not None:
        
            final_prediction = override

            # DEBUG ONLY
            # Remove this print later if desired.
            #print(
            #    f"\nOVERRIDE ACTIVE -> {override}"
            #)        
        # ==================================
        # TIMING
        # ==================================

        decision_timestamp = (
            datetime.utcnow()
            .isoformat() + "Z"
        )
        
        decision_latency_ms = int(
            (
                time.time()
                - trial_start
            )
            * 1000
        )

        # ==================================
        # BUILD RESULT
        # ==================================

        result = {

            "decision_timestamp":
                decision_timestamp,

            "decision_latency_ms":
                decision_latency_ms,

            "window_ms":
                2000,

            "gate_prediction":
                gate_label,

            "gate_confidence":
                float(
                    gate_confidence
                ),

            "mi_prediction":
                mi_label,

            "mi_confidence":
                float(
                    mi_confidence
                ),

            "final_prediction":
                final_prediction,

            "raw": {

                "gate_mi_votes":
                    int(
                        mi_gate_votes
                    ),

                "gate_ssvep_votes":
                    int(
                        ssvep_gate_votes
                    ),

                "left_votes":
                    int(
                        left_votes
                    ),

                "right_votes":
                    int(
                        right_votes
                    )
            }
        }

        print("\n================================")
        print("TRIAL COMPLETE")
        print("================================")

        print(
            json.dumps(
                result,
                indent=4
            )
        )

        return result
    # ======================================================
    # SAVE RESULTS
    # ======================================================

    def _save_json(
        self,
        result
    ):

        if not self.ENABLE_JSON_OUTPUT:

            return

        with open(
            self.JSON_FILE,
            "w"
        ) as f:

            json.dump(
                result,
                f,
                indent=4
            )

    # ======================================================
    # LOG RESULT
    # ======================================================

    def _log_result(
        self,
        result
    ):

        if not self.ENABLE_LOGGING:

            return

        file_exists = os.path.exists(
            self.LOG_FILE
        )

        with open(
            self.LOG_FILE,
            "a",
            newline=""
        ) as f:

            writer = csv.writer(f)

            if not file_exists:

                writer.writerow([

                    "decision_timestamp",

                    "decision_latency_ms",

                    "window_ms",

                    "gate_prediction",
                    "gate_confidence",

                    "mi_prediction",
                    "mi_confidence",

                    "final_prediction",

                    "gate_mi_votes",
                    "gate_ssvep_votes",

                    "left_votes",
                    "right_votes"
                ])

            writer.writerow([

                result[
                    "decision_timestamp"
                ],

                result[
                    "decision_latency_ms"
                ],

                result[
                    "window_ms"
                ],

                result[
                    "gate_prediction"
                ],

                result[
                    "gate_confidence"
                ],

                result[
                    "mi_prediction"
                ],

                result[
                    "mi_confidence"
                ],

                result[
                    "final_prediction"
                ],

                result[
                    "raw"
                ][
                    "gate_mi_votes"
                ],

                result[
                    "raw"
                ][
                    "gate_ssvep_votes"
                ],

                result[
                    "raw"
                ][
                    "left_votes"
                ],

                result[
                    "raw"
                ][
                    "right_votes"
                ]
            ])

    # ======================================================
    # RUN TRIAL WRAPPER
    # ======================================================

    def run_trial(self):

        result = self._run_trial_core()

        self._save_json(
            result
        )

        self._log_result(
            result
        )

        return result

    # ======================================================
    # SHUTDOWN
    # ======================================================

    def shutdown(self):

        print("\n================================")
        print("SHUTDOWN")
        print("================================")

        try:

            if self.inlet is not None:

                del self.inlet

        except:

            pass

        self.initialized = False
        self.presenter.shutdown()
        print(
            "Classifier shutdown complete."
        )


if __name__ == "__main__":

    classifier = HybridClassifier()

    classifier.initialize()

    result = classifier.run_trial()

    print("\nFINAL OUTPUT:")
    print(
        result[
            "final_prediction"
        ]
    )

    classifier.shutdown()