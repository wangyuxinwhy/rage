from typing import Self, Unpack

from rage.case import RageCase
from rage.metrics.base import GenerateBasedMetric
from rage.models import RageClassifier, RageClassifierKwargs
from rage.results import FaithfulnessResult

default_faithfulness_instruction = (
    "You are tasked to evaluate whether the generated answer is fully supported by the retrieved context."
)


class GenerateBasedAnswerFaithfulness(GenerateBasedMetric[FaithfulnessResult]):
    required_fields = {"question", "retrieved_contexts", "generated_answer"}

    model: RageClassifier

    def __init__(self, model: RageClassifier, label_score_mapping: dict[str, float]) -> None:
        super().__init__(model)
        self.label_score_mapping = label_score_mapping
        self._check()

    @classmethod
    def defaults(cls) -> Self:
        classifier = RageClassifier(
            instruction=default_faithfulness_instruction,
            label_set={"Yes", "No"},
        )
        label_score_mapping = {
            "Yes": 1.0,
            "No": 0.0,
        }
        return cls(classifier, label_score_mapping)

    @classmethod
    def from_parameters(
        cls, label_score_mapping: dict[str, float] | None = None, **kwargs: Unpack[RageClassifierKwargs]
    ) -> Self:
        default_classifier_kwargs = {
            "instruction": default_faithfulness_instruction,
            "label_set": {"Yes", "No"},
        }
        classifier_kwargs = {**default_classifier_kwargs, **kwargs}
        model = RageClassifier(**classifier_kwargs)
        label_score_mapping = label_score_mapping or {
            "Yes": 1.0,
            "No": 0.0,
        }
        return cls(model=model, label_score_mapping=label_score_mapping)

    def _check(self) -> None:
        for label in self.model.label_set:
            if label not in self.label_score_mapping:
                raise ValueError(f"Label '{label}' is not in label_score_mapping")

        for example in self.model.examples:
            if example.output.label not in self.label_score_mapping:
                raise ValueError(f"Label '{example.output.label}' is not in label_score_mapping")

    def calculate(self, case: RageCase) -> FaithfulnessResult:
        case = self.refine_case(case)
        classifier_output = self.model.inference(case)
        return FaithfulnessResult(
            faithfulness=self.label_score_mapping[classifier_output.label],
        )
