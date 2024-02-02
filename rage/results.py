from typing import Any

from pydantic import BaseModel


class RageResult(BaseModel):
    extra: dict[str, Any] = {}


class RelevanceResult(RageResult):
    relevance: float


class CoverageResult(RageResult):
    coverage: float


class PresicionResult(RageResult):
    precision: float
    average_precision: float
    precision_at_k: list[float]


class CorrectnessResult(RageResult):
    correctness: float


class FaithfulnessResult(RageResult):
    faithfulness: float


class PrecisionRecallF1Result(RageResult):
    precision: float
    recall: float
    f1: float
