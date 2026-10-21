from abc import ABC, abstractmethod
import random
import time


class IClassifier(ABC):

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def run_trial(self):
        pass

    @abstractmethod
    def shutdown(self):
        pass


class MockClassifier(IClassifier):

    def initialize(self):
        print("Classifier initialized.")

    def run_trial(self):

        print("Collecting EEG...")

        time.sleep(5)

        prediction = random.choice(
            [
                "LEFT",
                "RIGHT",
                "UP"
            ]
        )

        return {

            "decision_timestamp": str(time.time()),

            "decision_latency_ms": 5000,

            "window_ms": 2000,

            "gate_prediction": "MI",

            "gate_confidence": 0.8,

            "mi_prediction": prediction,

            "mi_confidence": 0.75,

            "final_prediction": prediction,

            "raw": {}
        }

    def shutdown(self):
        print("Classifier shutdown.")