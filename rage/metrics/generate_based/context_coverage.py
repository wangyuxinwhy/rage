from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, RootModel
from typing_extensions import Self, Unpack, override

from rage.case import RageCase
from rage.metrics.base import GenerateBasedMetric
from rage.models import RageModel, RageModelKwargs
from rage.results import CoverageResult

default_context_coverage_instruction = "Given a question, context, and answer, analyze each statement in the answer and classify if the statement can be attributed to the given context or not."


class AnswerStatement(BaseModel):
    statement: str
    reason: str
    supported: Literal["Yes", "No"]


AnswerStatements = RootModel[List[AnswerStatement]]


class GenerateBasedContextCoverage(GenerateBasedMetric[CoverageResult]):
    required_fields = {"question", "retrieved_contexts", "answer"}

    model: RageModel[AnswerStatements]

    def __init__(self, model: RageModel[AnswerStatements]) -> None:
        super().__init__(model)

    @classmethod
    def defaults(cls) -> Self:
        return cls(RageModel(instruction=default_context_coverage_instruction, output_structure=AnswerStatements))

    @classmethod
    def from_parameters(cls, **kwargs: Unpack[RageModelKwargs]) -> Self:
        default_extractor_kwargs = {
            "instruction": default_context_coverage_instruction,
            "output_structure": AnswerStatements,
        }
        extractor_kwargs = {**default_extractor_kwargs, **kwargs}
        if extractor_kwargs["output_structure"] != AnswerStatements:
            raise ValueError("The output type of the model must be AnswerStatements")
        return cls(model=RageModel(**extractor_kwargs))

    @override
    def calculate(self, case: RageCase) -> CoverageResult:
        if not case.retrieved_contexts:
            return CoverageResult(
                extra={"answer_statements": []},
                coverage=0,
            )
        case = self.refine_case(case)
        answer_statements = self.model.inference(case)
        coverage = sum([i.supported == "Yes" for i in answer_statements.root]) / len(answer_statements.root)
        return CoverageResult(
            extra={"answer_statements": answer_statements},
            coverage=coverage,
        )
