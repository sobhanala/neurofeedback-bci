"""Hybrid MI + SSVEP classifier integration for maze movement."""
import csv
import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

# CSP models trained under an older MNE / Python cannot be restored by the
# newer MNE class hierarchy (its __setstate__ requires keys like ``info`` and
# ``restr_type`` that the old pickles never stored, and calls _restore_callables
# which references them). Since we only ever call ``transform()`` on these
# objects, we bypass MNE entirely: a custom Unpickler swaps CSP/SPoC for a
# lightweight capture class that just stores the pickled state, then we wrap
# the captured ``filters_``/``n_components`` in a pure-numpy shim that
# reimplements transform() = log(mean((filters_[:n] @ X) ** 2)).


class _PlainCSP:
    """Pure-numpy stand-in for an MNE CSP object.

    Equivalent to MNE CSP with transform_into="average_power", log=True (the
    defaults these models were trained with).
    """

    def __init__(self, filters, n_components=None, log=True, mean_=None, std_=None):
        import numpy as np
        self._np = np
        self.filters_ = np.asarray(filters, dtype=np.float64)
        self.n_components = n_components or self.filters_.shape[0]
        self.log = log
        self.mean_ = np.asarray(mean_, dtype=np.float64) if mean_ is not None else None
        self.std_ = np.asarray(std_, dtype=np.float64) if std_ is not None else None

    def transform(self, X):
        np = self._np
        X = np.asarray(X, dtype=np.float64)
        pick = self.filters_[: self.n_components]
        projected = np.array([np.dot(pick, epoch) for epoch in X])
        power = (projected ** 2).mean(axis=2)
        if self.log:
            return np.log(power)
        if self.mean_ is not None and self.std_ is not None:
            return (power - self.mean_) / self.std_
        return power


class _CSPStateCapture:
    """Stand-in class used during unpickling to swallow the CSP state dict.

    MNE's newer CSP __init__/__setstate__ would raise on the old pickles, so we
    substitute this no-op class in find_class() and capture the state directly.
    """

    def __init__(self, *args, **kwargs):
        self._captured = {}

    def __setstate__(self, state):
        self._captured = dict(state) if isinstance(state, dict) else {}

    def __getstate__(self):
        return self._captured


def _csp_to_plain(captured):
    return _PlainCSP(
        filters=captured._captured.get("filters_"),
        n_components=captured._captured.get("n_components"),
        log=captured._captured.get("log", True),
        mean_=captured._captured.get("mean_"),
        std_=captured._captured.get("std_"),
    )


def _convert_csp_captures(obj):
    """Recursively replace _CSPStateCapture instances with _PlainCSP shims."""
    if isinstance(obj, _CSPStateCapture):
        return _csp_to_plain(obj)
    if isinstance(obj, list):
        return [_convert_csp_captures(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_convert_csp_captures(item) for item in obj)
    return obj


def _load_pickle_compat(file_obj):
    """Load a pickle; convert any old MNE CSP/SPoC objects to numpy-only shims.

    Non-CSP objects (sklearn Selector, SVM, Scaler, LDA) load normally.
    """
    import io

    class CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            cls = super().find_class(module, name)
            if "mne" in module and name in {"CSP", "SPoC"}:
                return _CSPStateCapture
            return cls

    obj = CompatUnpickler(io.BytesIO(file_obj.read())).load()
    return _convert_csp_captures(obj)


class HybridClassifierError(RuntimeError):
    """Raised when the hybrid classifier cannot initialize or predict."""


class HybridClassifier:
    MI_SUBJECT_MODEL = os.getenv("MI_SUBJECT_MODEL", "P201")
    FS = 256
    WINDOW_SAMPLES = 512
    UPDATE_SAMPLES = 64
    TRIAL_TIME = float(os.getenv("HYBRID_CLASSIFIER_TRIAL_TIME", "5.0"))
    LSL_STREAM_NAME = os.getenv("LSL_STREAM_NAME", "EmotivEEG")

    ENABLE_LOGGING = os.getenv("HYBRID_CLASSIFIER_LOGGING", "1") == "1"
    ENABLE_JSON_OUTPUT = os.getenv("HYBRID_CLASSIFIER_JSON", "1") == "1"
    LOG_FILE = os.getenv("HYBRID_CLASSIFIER_LOG_FILE", "prediction_log.csv")
    JSON_FILE = os.getenv("HYBRID_CLASSIFIER_JSON_FILE", "latest_prediction.json")

    def __init__(self, model_root=None):
        self.model_root = Path(
            model_root
            or os.getenv("FUSION_MODEL_DIR")
            or Path(__file__).resolve().parents[2] / "models" / "fusion"
        )
        self.initialized = False
        self.inlet = None
        self.csp_models = None
        self.selector = None
        self.svm = None
        self.mi_model_info = None
        self.scaler = None
        self.lda = None
        self.gate_model_info = None
        self._np = None
        self._butter = None
        self._filtfilt = None
        self._welch = None
        self._StreamInlet = None
        self._resolve_byprop = None

    def initialize(self):
        if self.initialized:
            return

        self._load_dependencies()
        self._load_mi_model()
        self._load_gate_model()
        self._connect_lsl()
        self.initialized = True

    def _load_dependencies(self):
        try:
            import numpy as np
            from pylsl import StreamInlet, resolve_byprop, resolve_streams
            from scipy.signal import butter, filtfilt, welch
        except ImportError as exc:
            raise HybridClassifierError(
                "Hybrid classifier dependencies are missing. Install numpy, scipy, pylsl, "
                "scikit-learn, and any package used to create the CSP pickle files."
            ) from exc

        self._np = np
        self._butter = butter
        self._filtfilt = filtfilt
        self._welch = welch
        self._StreamInlet = StreamInlet
        self._resolve_byprop = resolve_byprop
        self._resolve_streams = resolve_streams

    def _open_model_file(self, *parts, mode="rb"):
        path = self.model_root.joinpath(*parts)
        if not path.exists():
            raise HybridClassifierError(f"Required model file not found: {path}")
        return path.open(mode)

    def _load_mi_model(self):
        model_dir = ("final_models", self.MI_SUBJECT_MODEL)

        with self._open_model_file(*model_dir, "model_info.json", mode="r") as model_file:
            self.mi_model_info = json.load(model_file)
        with self._open_model_file(*model_dir, "csp_models.pkl") as model_file:
            self.csp_models = _load_pickle_compat(model_file)
        with self._open_model_file(*model_dir, "selector.pkl") as model_file:
            self.selector = _load_pickle_compat(model_file)
        with self._open_model_file(*model_dir, "svm.pkl") as model_file:
            self.svm = _load_pickle_compat(model_file)

    def _load_gate_model(self):
        with self._open_model_file("gate_model", "scaler.pkl") as model_file:
            self.scaler = _load_pickle_compat(model_file)
        with self._open_model_file("gate_model", "lda.pkl") as model_file:
            self.lda = _load_pickle_compat(model_file)
        with self._open_model_file("gate_model", "model_info.json", mode="r") as model_file:
            self.gate_model_info = json.load(model_file)

    def _resolve_any_eeg_stream(self, timeout=5):
        """Best-effort discovery of an EEG outlet when the configured name misses.

        Tries, in order: type == 'EEG', then any stream whose channel count
        matches the expected 14-channel Emotiv layout.
        """
        try:
            by_type = self._resolve_byprop("type", "EEG", timeout=timeout)
        except Exception:
            by_type = []
        if by_type:
            return by_type[0]

        try:
            all_streams = self._resolve_streams(wait_time=timeout) if self._resolve_streams else []
        except Exception:
            all_streams = []
        for stream in all_streams:
            try:
                if stream.channel_count() == 14:
                    return stream
            except Exception:
                continue
        return None

    def _connect_lsl(self):
        """Wait for the EEG stream like the working standalone classifier.

        First resolves by the configured name; if that misses, falls back to
        any EEG-type or 14-channel stream so it connects regardless of the
        outlet name the headset publishes.
        """
        while True:
            streams = self._resolve_byprop("name", self.LSL_STREAM_NAME, timeout=2)
            if streams:
                self.inlet = self._StreamInlet(streams[0])
                return

            fallback = self._resolve_any_eeg_stream(timeout=2)
            if fallback is not None:
                self.inlet = self._StreamInlet(fallback)
                return

            time.sleep(1)

    def bandpass_epoch(self, epoch, low, high, fs):
        nyquist = fs / 2
        b, a = self._butter(4, [low / nyquist, high / nyquist], btype="band")
        return self._filtfilt(b, a, epoch, axis=1)

    def gate_bandpass(self, epoch):
        nyquist = self.FS / 2
        b, a = self._butter(4, [8 / nyquist, 45 / nyquist], btype="band")
        return self._filtfilt(b, a, epoch, axis=1)

    def extract_mi_features(self, epoch):
        np = self._np
        epoch = (epoch - epoch.mean(axis=1, keepdims=True)) / (
            epoch.std(axis=1, keepdims=True) + 1e-8
        )

        feature_blocks = []
        bands = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 30)]

        for (low, high), csp in zip(bands, self.csp_models):
            epoch_band = self.bandpass_epoch(epoch, low, high, self.FS)
            csp_features = csp.transform(epoch_band[np.newaxis, :, :])
            feature_blocks.append(csp_features)

        features = np.concatenate(feature_blocks, axis=1)
        return self.selector.transform(features)

    def extract_gate_features(self, epoch):
        np = self._np
        epoch = self.gate_bandpass(epoch)
        epoch = (epoch - epoch.mean(axis=1, keepdims=True)) / (
            epoch.std(axis=1, keepdims=True) + 1e-8
        )

        features = []
        bands = [(4, 8), (8, 13), (13, 30), (30, 45)]

        for channel_index in range(epoch.shape[0]):
            freqs, psd = self._welch(epoch[channel_index], fs=self.FS, nperseg=256)
            for low, high in bands:
                band_index = (freqs >= low) & (freqs < high)
                features.append(np.mean(psd[band_index]))

        return np.array(features)

    def run_trial(self):
        if not self.initialized:
            self.initialize()

        result = self._run_trial_core()

        self._save_json(result)
        self._log_result(result)
        return result

    def _run_trial_core(self):
        np = self._np
        buffer = np.empty((14, 0), dtype=np.float32)
        samples_since_prediction = 0
        mi_predictions = []
        mi_probabilities = []
        gate_predictions = []
        gate_probabilities = []
        trial_start = time.time()

        while time.time() - trial_start < self.TRIAL_TIME:
            sample, _timestamp = self.inlet.pull_sample()
            sample = np.array(sample, dtype=np.float32).reshape(14, 1)
            buffer = np.concatenate([buffer, sample], axis=1)

            if buffer.shape[1] > self.WINDOW_SAMPLES:
                buffer = buffer[:, -self.WINDOW_SAMPLES:]
            if buffer.shape[1] < self.WINDOW_SAMPLES:
                continue

            samples_since_prediction += 1
            if samples_since_prediction < self.UPDATE_SAMPLES:
                continue
            samples_since_prediction = 0

            current_window = buffer.copy()

            mi_features = self.extract_mi_features(current_window)
            mi_pred = self.svm.predict(mi_features)[0]
            mi_probs = self.svm.predict_proba(mi_features)[0]
            mi_predictions.append(mi_pred)
            mi_probabilities.append(mi_probs)

            gate_features = self.extract_gate_features(current_window)
            gate_features = self.scaler.transform([gate_features])
            gate_pred = self.lda.predict(gate_features)[0]
            gate_probs = self.lda.predict_proba(gate_features)[0]
            gate_predictions.append(gate_pred)
            gate_probabilities.append(gate_probs)

        if not mi_predictions:
            raise HybridClassifierError("No predictions generated from EEG samples.")

        mi_predictions = np.array(mi_predictions)
        mi_probabilities = np.array(mi_probabilities)
        left_votes = np.sum(mi_predictions == 0)
        right_votes = np.sum(mi_predictions == 1)
        avg_mi_probs = np.mean(mi_probabilities, axis=0)

        if left_votes > right_votes:
            mi_label = "LEFT"
            mi_confidence = avg_mi_probs[0]
        else:
            mi_label = "RIGHT"
            mi_confidence = avg_mi_probs[1]

        gate_predictions = np.array(gate_predictions)
        gate_probabilities = np.array(gate_probabilities)
        mi_gate_votes = np.sum(gate_predictions == 0)
        ssvep_gate_votes = np.sum(gate_predictions == 1)
        avg_gate_probs = np.mean(gate_probabilities, axis=0)

        if mi_gate_votes > ssvep_gate_votes:
            gate_label = "MI"
            gate_confidence = avg_gate_probs[0]
        else:
            gate_label = "SSVEP"
            gate_confidence = avg_gate_probs[1]

        final_prediction = "UP" if gate_label == "SSVEP" else mi_label

        decision_latency_ms = int((time.time() - trial_start) * 1000)

        return {
            "decision_timestamp": datetime.utcnow().isoformat() + "Z",
            "decision_latency_ms": decision_latency_ms,
            "window_ms": 2000,
            "gate_prediction": gate_label,
            "gate_confidence": float(gate_confidence),
            "mi_prediction": mi_label,
            "mi_confidence": float(mi_confidence),
            "final_prediction": final_prediction,
            "eeg": {
                "attention": float(mi_confidence),
                "cognitive_load": 1.0 - float(gate_confidence),
                "relaxation": 1.0 - float(mi_confidence),
            },
            "raw": {
                "gate_mi_votes": int(mi_gate_votes),
                "gate_ssvep_votes": int(ssvep_gate_votes),
                "left_votes": int(left_votes),
                "right_votes": int(right_votes),
            },
        }

    def _save_json(self, result):
        if not self.ENABLE_JSON_OUTPUT:
            return
        with open(self.JSON_FILE, "w") as output_file:
            json.dump(result, output_file, indent=4)

    def _log_result(self, result):
        if not self.ENABLE_LOGGING:
            return

        file_exists = os.path.exists(self.LOG_FILE)
        with open(self.LOG_FILE, "a", newline="") as log_file:
            writer = csv.writer(log_file)
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
                    "right_votes",
                ])

            writer.writerow([
                result["decision_timestamp"],
                result["decision_latency_ms"],
                result["window_ms"],
                result["gate_prediction"],
                result["gate_confidence"],
                result["mi_prediction"],
                result["mi_confidence"],
                result["final_prediction"],
                result["raw"]["gate_mi_votes"],
                result["raw"]["gate_ssvep_votes"],
                result["raw"]["left_votes"],
                result["raw"]["right_votes"],
            ])

    def shutdown(self):
        if self.inlet is not None:
            del self.inlet
        self.inlet = None
        self.initialized = False


_classifier = None


def get_hybrid_classifier():
    global _classifier
    if _classifier is None:
        model_root = Path(
            os.getenv("FUSION_MODEL_DIR")
            or Path(__file__).resolve().parents[2] / "models" / "fusion"
        )
        _classifier = HybridClassifier(model_root=model_root)
    return _classifier

def prediction_to_maze_action(prediction):
    return {
        "LEFT": "move_left",
        "RIGHT": "move_right",
        "UP": "move_up",
    }.get(prediction)
