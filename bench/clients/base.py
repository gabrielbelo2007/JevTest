from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

@dataclass
class Prediction:
    case_id: str
    task: str
    model: str
    ok: bool
    error: str | None
    pred: str | int | bool | None
    probs: dict[str, float] | None
    p_true: float | None
    score_value: float | None
    latency_ms: float
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
