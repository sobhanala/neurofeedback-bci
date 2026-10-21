from dataclasses import dataclass
from typing import Dict


@dataclass
class DecisionResult:

    decision_timestamp: str

    decision_latency_ms: int

    window_ms: int

    gate_prediction: str

    gate_confidence: float

    mi_prediction: str

    mi_confidence: float

    final_prediction: str

    raw: Dict

@dataclass
class CommandMessage:

    command_id: int

    command: str

    timestamp: str

@dataclass
class AckMessage:

    command_id: int

    timestamp: str