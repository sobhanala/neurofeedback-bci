from hybrid_classifier import HybridClassifier
from state_machine import FusionState
from unity_tcp import UnityTCPServer
from experiment_logger import ExperimentLogger
from datetime import datetime
import time


class FusionEngine:

    def __init__(self):

        self.classifier = HybridClassifier()

        self.unity = UnityTCPServer()

        self.state = FusionState.STARTUP

        self.trial_id = 0

        self.command_id = 0

        self.logger = ExperimentLogger()
    def prediction_to_command(self, prediction):

        mapping = {
            "LEFT": "MOVE_LEFT",
            "RIGHT": "MOVE_RIGHT",
            "UP": "MOVE_UP"
        }

        return mapping.get(prediction)

    def run(self):

        print("Initializing classifier...")

        self.classifier.initialize()

        print("Waiting for Unity connection...")

        self.unity.start()

        print("Fusion Engine Ready.")

        self.state = FusionState.READY

        try:
            while self.state != FusionState.SHUTDOWN:

                self.trial_id += 1

                print()
                print("=" * 50)
                print(f"TRIAL {self.trial_id}")
                print("=" * 50)

                self.state = FusionState.RUNNING_TRIAL

                print("Running classifier...")

                result = self.classifier.run_trial()

                prediction = result["final_prediction"]

                print(f"Prediction: {prediction}")

                command = self.prediction_to_command(
                    prediction
                )

                print(f"Command: {command}")

                self.command_id += 1

                fusion_send_timestamp = (
                    datetime.utcnow().isoformat() + "Z"
                )
                

                message = {

                    "schema": "fusion.unity",

                    "schema_version": "1.0",

                    "msg_type": "COMMAND",

                    "command_id": self.command_id,

                    "decision_timestamp":
                        result["decision_timestamp"],

                    "fusion_send_timestamp":
                        fusion_send_timestamp,

                    "payload": {
                        "command": command
                    }
                }

                self.unity.send_json(message)

                print(
                    f"Command sent "
                    f"(ID={self.command_id})"
                )

                self.state = (
                    FusionState.WAITING_ACK
                )

                ack = self.wait_for_ack(
                    self.command_id
                )
                
                if not ack:

                    self.logger.log_trial(

                        trial_id=self.trial_id,

                        prediction=prediction,

                        command=command,

                        classifier_latency_ms=
                        result["decision_latency_ms"],

                        decision_timestamp=
                        result["decision_timestamp"],

                        fusion_send_timestamp=
                        fusion_send_timestamp,

                        unity_receive_timestamp="",

                        unity_execute_timestamp="",

                        fusion_ack_receive_timestamp="",

                        status="TIMEOUT"
                    )
                

                    continue
                fusion_ack_receive_timestamp = (
                datetime.utcnow().isoformat() + "Z"
            )
                unity_receive_timestamp = ack.get(
                    "unity_receive_timestamp",
                    ""
                )

                unity_execute_timestamp = ack.get(
                    "unity_execute_timestamp",
                    ""
                )

                self.state = (
                    FusionState.REFRACTORY
                )

                print()

                print(
                    "Refractory period..."
                )
                self.logger.log_trial(

                    trial_id=self.trial_id,

                    prediction=prediction,

                    command=command,

                    classifier_latency_ms=
                    result["decision_latency_ms"],

                    decision_timestamp=
                    result["decision_timestamp"],

                    fusion_send_timestamp=
                    fusion_send_timestamp,

                    unity_receive_timestamp=
                    unity_receive_timestamp,

                    unity_execute_timestamp=
                    unity_execute_timestamp,

                    fusion_ack_receive_timestamp=
                    fusion_ack_receive_timestamp,

                    status="SUCCESS"
                )

                time.sleep(3)
        except KeyboardInterrupt:
            self.shutdown()     


    def wait_for_ack(
        self,
        expected_command_id
    ):
        
        print(
            f"Waiting for ACK "
            f"{expected_command_id}"
        )

        ack = self.unity.receive_json()
        if ack == "TIMEOUT":

            print(
                "ACK timeout."
            )

            return False
        
        if ack is None:

            print(
                "Connection lost."
            )

            return False

        if ack["msg_type"] != "ACK":

            print(
                "Invalid message."
            )

            return False

        if (
            ack["command_id"]
            != expected_command_id
        ):

            print(
                "Wrong ACK."
            )

            return False

        print(
            f"ACK received "
            f"for command "
            f"{expected_command_id}"
        )

        return ack
    
    def shutdown(self):

        self.state = FusionState.SHUTDOWN

        print()
        print("=" * 50)
        print("Fusion shutting down...")
        print("=" * 50)

        try:
            print("Closing logger...")
            self.logger.close()

        except Exception as e:
            print(f"Logger close error: {e}")

        try:
            print("Closing Unity sockets...")
            self.unity.close()

        except Exception as e:
            print(f"Socket close error: {e}")

        try:
            print("Shutting down classifier...")
            self.classifier.shutdown()

        except Exception as e:
            print(f"Classifier shutdown error: {e}")

        print()
        print("Shutdown complete.")