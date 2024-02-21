from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel


class RageResult(BaseModel):
    extra: Dict[str, Any] = {}


class RelevanceResult(RageResult):
    relevance: float


class CoverageResult(RageResult):
    coverage: float


class PresicionResult(RageResult):
    precision: float
    average_precision: float
    precision_at_k: List[float]


class CorrectnessResult(RageResult):
    correctness: float


class FaithfulnessResult(RageResult):
    faithfulness: float


class PrecisionRecallF1Result(RageResult):
    precision: float
    recall: float
    f1: float
