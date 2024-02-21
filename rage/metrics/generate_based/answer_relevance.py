from typing_extensions import Self, Unpack

from rage.case import RageCase
from rage.metrics.base import GenerateBasedMetric
from rage.models import RageScorer, RageScorerKwargs
from rage.results import RelevanceResult

default_answer_relevance_instruction = """You are an expert evaluator system for a question answering system.
You need to evaluate the relevance and completeness of the generated answer based on the question.
Use the following guidelines for evaluation:
* score between 1 to 3.
* 1 means that the answer is completely irrelevant to the question.
* 2 means that the answer is partially relevant to the question or it only partially answers the question.
* 3 means that the answer is relevant to the question and completely answers the question.
"""


class GenerateBasedAnswerRelevance(GenerateBasedMetric[RelevanceResult]):
    required_fields = {"question", "generated_answer"}

    model: RageScorer

    def __init__(self, model: RageScorer) -> None:
        super().__init__(model)

    @classmethod
    def defaults(cls) -> Self:
        return cls(RageScorer(instruction=default_answer_relevance_instruction, score_range=(1, 3)))

    @classmethod
    def from_parameters(cls, **kwargs: Unpack[RageScorerKwargs]) -> Self:
        default_scorer_kwargs = {
            "instruction": default_answer_relevance_instruction,
            "score_range": (1, 3),
        }
        scorer_kwargs = {**default_scorer_kwargs, **kwargs}
        return cls(model=RageScorer(**scorer_kwargs))

    def calculate(self, case: RageCase) -> RelevanceResult:
        case = self.refine_case(case)
        score_result = self.model.inference(case)
        return RelevanceResult(relevance=score_result.score, extra={"reason": getattr(score_result, "reason", None)})
