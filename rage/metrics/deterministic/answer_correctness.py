from rage.case import RageCase
from rage.metrics.base import RageMetric
from rage.results import CorrectnessResult
from rage.utils import caculate_rouge_l_score, calculate_word_overlap


class WorldOverlapAnswerCorrectness(RageMetric[CorrectnessResult]):
    required_fields = {"answer", "generated_answer"}

    def calculate(self, case: RageCase) -> CorrectnessResult:
        case = self.refine_case(case)
        correctness = calculate_word_overlap(case.answer, case.generated_answer).f1
        return CorrectnessResult(correctness=correctness)


class RougeLAnswerCorrectness(RageMetric[CorrectnessResult]):
    required_fields = {"answer", "generated_answer"}

    def calculate(self, case: RageCase) -> CorrectnessResult:
        case = self.refine_case(case)
        rouge_l_score = caculate_rouge_l_score(case.answer, case.generated_answer).f1
        return CorrectnessResult(correctness=rouge_l_score)
