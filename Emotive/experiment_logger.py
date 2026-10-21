import csv
import os


class ExperimentLogger:

    def __init__(
        self,
        filename="experiment_log.csv"
    ):

        self.filename = filename

        file_exists = os.path.exists(
            filename
        )

        self.file = open(
            filename,
            "a",
            newline=""
        )

        self.writer = csv.writer(
            self.file
        )

        if not file_exists:

            self.writer.writerow([
                "trial_id",

                "prediction",

                "command",

                "classifier_latency_ms",

                "decision_timestamp",

                "fusion_send_timestamp",

                "unity_receive_timestamp",

                "unity_execute_timestamp",

                "fusion_ack_receive_timestamp",

                "status"
            ])
    def log_trial(
        self,
        trial_id,
        prediction,
        command,
        classifier_latency_ms,
        decision_timestamp,
        fusion_send_timestamp,
        unity_receive_timestamp,
        unity_execute_timestamp,
        fusion_ack_receive_timestamp,
        status
    ):

        self.writer.writerow([
            trial_id,

            prediction,

            command,

            classifier_latency_ms,

            decision_timestamp,

            fusion_send_timestamp,

            unity_receive_timestamp,

            unity_execute_timestamp,

            fusion_ack_receive_timestamp,

            status
        ])

        self.file.flush()

    def close(self):

        self.file.close()