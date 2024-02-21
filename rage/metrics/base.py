from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

from rage.case import RageCase
from rage.models import RageModel
from rage.results import RageResult

T = TypeVar("T", bound=RageResult)


class RageMetric(Generic[T], ABC):
    required_fields: ClassVar[set[str]] = set()
    optional_fields: ClassVar[set[str]] = set()

    @abstractmethod
    def calculate(self, case: RageCase) -> T:
        ...

    def refine_case(self, case: RageCase) -> RageCase:
        values = {}
        for field in self.required_fields:
            if getattr(case, field) is None:
                raise ValueError(f"Field '{field}' is required")
            values[field] = getattr(case, field)
        for field in self.optional_fields:
            values[field] = getattr(case, field)
        return RageCase(**values)


class GenerateBasedMetric(RageMetric[T], ABC):
    def __init__(self, model: RageModel) -> None:
        self.model = model
        for example in self.model.examples:
            example.rage_case = self.refine_case(example.rage_case)
