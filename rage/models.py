import json
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Literal, Type, TypedDict

from generate import load_chat_model
from generate.chat_completion import ChatCompletionModel
from generate.chat_completion.base import RemoteChatCompletionModel
from generate.chat_completion.message import (
    AssistantMessage,
    SystemMessage,
    UnionMessage,
    UserMessage,
)
from pydantic import BaseModel, Field, create_model

from rage.case import RageCase, RageExample
from rage.template import RageCaseTemplate, SimpleCaseTemplate
from rage.utils import ensure_valid_json

cot_reason_description = "Think Step by Step, [!IMPORTANT] First write your reason and then give a score. [!IMPORTANT]"
field_info_title = "Output JSON strictly based Pydantic FieldInfo and MUST generate by field order:\n"
json_schema_title = "Output JSON strictly based OpenAI JSON Schema:\n"


def generate_structure[T: BaseModel](
    chat_model: ChatCompletionModel, prompt: list[UnionMessage], structure_type: Type[T], max_num_reask: int = 2
) -> T:
    num_reask = 0
    prompt = deepcopy(prompt)

    while num_reask <= max_num_reask:
        model_output = chat_model.generate(prompt)
        prompt.append(model_output.message)
        try:
            json_string = ensure_valid_json(model_output.reply)
            return structure_type.model_validate_json(json_string)
        except Exception as e:
            num_reask += 1
            prompt.append(UserMessage(content=f"I got an error, please try to fix your output based on the error message. Error: {e}"))

    raise ValueError(f"Failed to generate valid JSON after {max_num_reask} reasks.")


class RageModelKwargs[T: BaseModel](TypedDict, total=False):
    instruction: str
    examples: list[RageExample[T]]
    model_id: str
    temperature: float
    timeout: int
    case_template: RageCaseTemplate


class RageModel[T: BaseModel](BaseModel, ABC):
    instruction: str
    examples: list[RageExample[T]] = []
    case_template: RageCaseTemplate = Field(default_factory=SimpleCaseTemplate)
    model_id: str = "openai/gpt-4-turbo-preview"
    temperature: float = 0.0
    timeout: int = 120

    @property
    def prompt(self) -> list[UnionMessage]:
        messages = []
        system_content = self.instruction + "\n" + self.output_format_section
        system_message = SystemMessage(content=system_content)
        messages.append(system_message)
        for example in self.examples:
            messages.append(UserMessage(content=self.case_template.format(example.rage_case)))
            messages.append(AssistantMessage(content=example.output.model_dump_json(exclude_none=True)))
        return messages

    @property
    def chat_model(self) -> ChatCompletionModel:
        if not hasattr(self, "_chat_model") or self._chat_model.model_id != self.model_id:
            self._chat_model = load_chat_model(self.model_id)
            if isinstance(self._chat_model, RemoteChatCompletionModel):
                self._chat_model.http_client.timeout = self.timeout
                self._chat_model.parameters.model_update(temperature=self.temperature)
        return self._chat_model

    @abstractmethod
    def inference(self, case: RageCase) -> T:
        ...

    @property
    def output_format_section(self) -> str:
        return ""


class ScorerOutput(BaseModel):
    reason: str | None = Field(
        None,
        description="According to the user's request, determine whether it is necessary to output the reason for the score. If the reason for the score needs to be output, it must be output first, followed by the score.",
    )
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

    def inference(self, case: RageCase) -> ScorerOutput:
        prompt = self.prompt
        prompt.append(UserMessage(content=self.case_template.format(case)))
        scorer_output = generate_structure(self.chat_model, prompt, ScorerOutput)
        if self.normalize:
            scorer_output.score = (scorer_output.score - self.score_range[0]) / (self.score_range[1] - self.score_range[0])
        return scorer_output

    @property
    def output_format_section(self) -> str:
        fields: dict[str, Any] = {}
        if self.cot:
            fields["reason"] = (str, Field(description=cot_reason_description))
        fields["score"] = (float, Field(ge=self.score_range[0], le=self.score_range[1]))
        output_pydantic_model = create_model("ScorerOutput", **fields)
        text = field_info_title
        field_info_text = ""
        for field, field_info in output_pydantic_model.model_fields.items():
            field_info_text += f"{field}: {field_info}\n"
        text += field_info_text
        return text


class ClassifierOutput(BaseModel):
    reason: str | None = None
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

    def inference(self, case: RageCase) -> ClassifierOutput:
        prompt = self.prompt
        prompt.append(UserMessage(content=self.case_template.format(case)))
        structure = generate_structure(self.chat_model, prompt, self.output_model_type)
        return ClassifierOutput(**structure.model_dump())

    @property
    def output_model_type(self) -> type[BaseModel]:
        fields = {}
        if self.cot:
            fields["reason"] = (str, Field(description=cot_reason_description))
        fields["label"] = (Literal[*tuple(self.label_set)], ...) # type: ignore
        output_pydantic_model = create_model("ClassifierOutput", **fields)
        return output_pydantic_model

    @property
    def output_format_section(self) -> str:
        output_pydantic_model = self.output_model_type

        text = json_schema_title
        json_schema = json.dumps(output_pydantic_model.model_json_schema(), indent=2)
        text += json_schema
        return text

        # text = field_info_title
        # field_info_text = ""
        # for field, field_info in output_pydantic_model.model_fields.items():
        #     field_info_text += f"{field}: {field_info}\n"
        # text += field_info_text
        # return text


class RageExtractorKwargs[T: BaseModel](RageModelKwargs[T], total=False):
    output_type: Type[T]


class RageExtractor[T: BaseModel](RageModel[T]):
    output_type: Type[T]

    def model_post_init(self, __context: Any) -> None:
        for example in self.examples:
            if not isinstance(example.output, self.output_type):
                raise TypeError(f"Output of example is not of type {self.output_type}")

    def inference(self, case: RageCase) -> T:
        prompt = self.prompt
        prompt.append(UserMessage(content=self.case_template.format(case)))
        return generate_structure(self.chat_model, prompt, self.output_type)

    @property
    def output_format_section(self) -> str:
        output_pydantic_model = self.output_type
        text = json_schema_title
        json_schema = json.dumps(output_pydantic_model.model_json_schema(), indent=2)
        text += json_schema
        return text
