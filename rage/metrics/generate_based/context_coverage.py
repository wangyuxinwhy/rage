from typing import Literal, Self, Unpack, override

from pydantic import BaseModel, RootModel

from rage.case import RageCase
from rage.metrics.base import GenerateBasedMetric
from rage.models import RageExtractor, RageExtractorKwargs
from rage.results import CoverageResult

default_context_coverage_instruction = "Given a question, context, and answer, analyze each statement in the answer and classify if the statement can be attributed to the given context or not."


class AnswerStatement(BaseModel):
    statement: str
    reason: str
    attributed: Literal["No", "Yes"]


AnswerStatements = RootModel[list[AnswerStatement]]


class GenerateBasedContextCoverage(GenerateBasedMetric[CoverageResult]):
    required_fields = {"question", "retrieved_contexts", "answer"}

    model: RageExtractor[AnswerStatements]

    def __init__(self, model: RageExtractor[AnswerStatements]) -> None:
        super().__init__(model)

    @classmethod
    def defaults(cls) -> Self:
        return cls(RageExtractor(instruction=default_context_coverage_instruction, output_type=AnswerStatements))

    @classmethod
    def from_parameters(cls, **kwargs: Unpack[RageExtractorKwargs]) -> Self:
        default_extractor_kwargs = {
            "instruction": default_context_coverage_instruction,
            "output_type": AnswerStatements,
        }
        extractor_kwargs = {**default_extractor_kwargs, **kwargs}
        if extractor_kwargs["output_type"] != AnswerStatements:
            raise ValueError("The output type of the model must be AnswerStatements")
        return cls(model=RageExtractor(**extractor_kwargs))

    @override
    def calculate(self, case: RageCase) -> CoverageResult:
        case = self.refine_case(case)
        answer_statements = self.model.inference(case)
        coverage = sum([i.attributed == "Yes" for i in answer_statements.root]) / len(answer_statements.root)
        return CoverageResult(
            extra={"answer_statements": answer_statements},
            coverage=coverage,
        )
