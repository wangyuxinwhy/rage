from typing_extensions import Self, Unpack

from rage.case import RageCase
from rage.metrics.base import GenerateBasedMetric
from rage.models import RageScorer, RageScorerKwargs
from rage.results import CorrectnessResult

default_correctness_instruction = """You are an expert evaluator system for a question answering system.
You need to evaluate the quality of the generated answer based on the question and reference ground truth answer.
Output concise reasons and scores.
Use the following guidelines for evaluation:
* You should output a single score between 1 to 5.
* 1 means that the answer is completely irrelevant to the question.
* 2 means that the answer is relevant to the question but contains major errors.
* 3 means that the answer is relevant to the question and is partially correct.
* 4 means that the answer is relevant to the question and is correct.
* 5 means that the answer is relevant to the question and is correct and complete.
"""


class GenerateBasedAnswerCorrectness(GenerateBasedMetric[CorrectnessResult]):
    required_fields = {"question", "answer", "generated_answer"}

    model: RageScorer

    def __init__(self, model: RageScorer) -> None:
        super().__init__(model)

    @classmethod
    def defaults(cls) -> Self:
        return cls(RageScorer(instruction=default_correctness_instruction, score_range=(1, 5)))

    @classmethod
    def from_parameters(cls, **kwargs: Unpack[RageScorerKwargs]) -> Self:
        default_scorer_kwargs = {
            "instruction": default_correctness_instruction,
            "score_range": (1, 5),
        }
        scorer_kwargs = {**default_scorer_kwargs, **kwargs}
        return cls(model=RageScorer(**scorer_kwargs))

    def calculate(self, case: RageCase) -> CorrectnessResult:
        case = self.refine_case(case)
        score_result = self.model.inference(case)
        return CorrectnessResult(correctness=score_result.score, extra={"reason": getattr(score_result, "reason", None)})
