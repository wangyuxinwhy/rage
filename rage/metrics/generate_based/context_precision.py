from typing import Self, Unpack

from rage.case import RageCase
from rage.metrics.base import GenerateBasedMetric
from rage.models import RageClassifier, RageClassifierKwargs
from rage.results import PresicionResult

default_context_precision_instruction = """\
Verify if the information in the given context is useful in answering the question.
if the information is useful, label is "Yes", otherwise "No".
"""


class GenerateBasedContextPrecision(GenerateBasedMetric[PresicionResult]):
    required_fields = {"question", "retrieved_contexts"}

    model: RageClassifier

    def __init__(self, model: RageClassifier) -> None:
        if model.label_set != {"Yes", "No"}:
            raise ValueError("The label set of the model must be {'Yes', 'No'}")
        super().__init__(model)

    @classmethod
    def defaults(cls) -> Self:
        classifier = RageClassifier(
            instruction=default_context_precision_instruction,
            label_set={"Yes", "No"},
        )
        return cls(classifier)

    @classmethod
    def from_parameters(cls, **kwargs: Unpack[RageClassifierKwargs]) -> Self:
        default_classifier_kwargs = {
            "instruction": default_context_precision_instruction,
            "label_set": {"Yes", "No"},
        }
        classifier_kwargs = {**default_classifier_kwargs, **kwargs}
        return cls(model=RageClassifier(**classifier_kwargs))

    def calculate(self, case: RageCase) -> PresicionResult:
        case = self.refine_case(case)
        verifications: list = []
        num_positive = 0
        precision_at_k: list[float] = []
        for num_context, context in enumerate(case.retrieved_contexts, start=1):
            new_case = case.model_copy(deep=True)
            new_case.retrieved_contexts = [context]
            vertification = self.model.inference(new_case)
            verifications.append(vertification)
            if vertification.label == "Yes":
                num_positive += 1
            precision_at_k.append(num_positive / num_context)
        precision = num_positive / len(case.retrieved_contexts)
        average_precision = sum(precision_at_k) / len(precision_at_k)
        return PresicionResult(
            precision_at_k=precision_at_k,
            average_precision=average_precision,
            precision=precision,
            extra={"verifications": verifications},
        )
