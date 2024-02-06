from abc import ABC
from typing import Any, Literal, Type, TypedDict, override

from generate import load_chat_model
from generate.chat_completion.base import RemoteChatCompletionModel
from generate.chat_completion.message import (
    UserMessage,
)
from generate.modifiers.structure import Example, Structure
from pydantic import BaseModel, Field, create_model

from rage.case import RageCase, RageExample
from rage.template import RageCaseTemplate, SimpleCaseTemplate

cot_reason_description = "Think Step by Step, [!IMPORTANT] First write your reason and then give a score. [!IMPORTANT]"
system_template = """\
# Instruction
{instruction}
# Output Format
{output_format_description}
"""


class RageModelKwargs[T: BaseModel](TypedDict, total=False):
    instruction: str
    examples: list[RageExample[T]]
    model_id: str
    temperature: float
    timeout: int
    case_template: RageCaseTemplate
    system_template: str
    output_structure: Type[T] | None


class RageModel[T: BaseModel](BaseModel, ABC):
    instruction: str
    examples: list[RageExample[T]] = []
    case_template: RageCaseTemplate = Field(default_factory=SimpleCaseTemplate)
    model_id: str = "openai/gpt-4-turbo-preview"
    temperature: float = 0.01
    timeout: int = 120
    system_template: str = system_template
    output_structure: Type[T] | None = None

    @property
    def structure_model(self) -> Structure[Any, T]:
        if not hasattr(self, "_structure_model") or self._structure_model.model.model_id != self.model_id:
            _chat_model = load_chat_model(self.model_id)
            if isinstance(_chat_model, RemoteChatCompletionModel):
                _chat_model.http_client.timeout = self.timeout
                _chat_model.parameters.model_update(temperature=self.temperature)
            examples = [
                Example(prompt=self.case_template.format(example.rage_case), output=example.output) for example in self.examples
            ]
            self._structure_model = _chat_model.structure(self.instruction, self.output_pydantic_model, examples=examples)
        return self._structure_model

    @property
    def output_pydantic_model(self):
        if self.output_structure is not None:
            return self.output_structure
        raise NotImplementedError

    def inference(self, case: RageCase) -> T:
        return self.structure_model.generate(UserMessage(content=self.case_template.format(case))).structure


class ScorerOutput(BaseModel):
    score: float


class RageScorerKwargs(RageModelKwargs[ScorerOutput], total=False):
    score_range: tuple[float, float]
    normalize: bool
    cot: bool


class RageScorer(RageModel[ScorerOutput]):
    score_range: tuple[float, float]
    normalize: bool = True
    cot: bool = False

    def model_post_init(self, __context: Any) -> None:
        for example in self.examples:
            if example.output.score < self.score_range[0] or example.output.score > self.score_range[1]:
                raise ValueError(f"Score {example.output.score} is not in the range {self.score_range}")

    @override
    def inference(self, case: RageCase) -> ScorerOutput:
        scorer_output = super().inference(case)
        if self.normalize:
            scorer_output.score = (scorer_output.score - self.score_range[0]) / (self.score_range[1] - self.score_range[0])
        return scorer_output

    @property
    @override
    def output_pydantic_model(self) -> Type[ScorerOutput]:
        fields: dict[str, Any] = {}
        if self.cot:
            fields["reason"] = (str, Field(description=cot_reason_description))
        fields["score"] = (float, Field(ge=self.score_range[0], le=self.score_range[1]))
        return create_model("ScorerOutput", **fields)  # type: ignore


class ClassifierOutput(BaseModel):
    label: str


class RageClassifierKwargs(RageModelKwargs[ClassifierOutput], total=False):
    label_set: set[str]
    cot: bool


class RageClassifier(RageModel[ClassifierOutput]):
    label_set: set[str]
    cot: bool = False

    def model_post_init(self, __context: Any) -> None:
        for example in self.examples:
            if example.output.label not in self.label_set:
                raise ValueError(f"Label '{example.output.label}' is not in label_set")

    @property
    @override
    def output_pydantic_model(self) -> Type[ClassifierOutput]:
        fields = {}
        if self.cot:
            fields["reason"] = (str, Field(description=cot_reason_description))
        fields["label"] = (Literal[*tuple(self.label_set)], ...)  # type: ignore
        return create_model("ClassifierOutput", **fields)  # type: ignore
