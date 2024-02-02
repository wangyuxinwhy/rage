from abc import ABC, abstractmethod

from pydantic import BaseModel

from rage.case import RageCase


class RageCaseTemplate(BaseModel, ABC):
    @abstractmethod
    def format(self, case: RageCase) -> str:
        ...


class SimpleCaseTemplate(RageCaseTemplate):
    def format(self, case: RageCase) -> str:
        text = ""
        if case.question:
            text += "Question: " + case.question + "\n"
        if case.contexts:
            text += "Ground Truth Context: " + "\n".join(case.contexts) + "\n"
        if case.answer:
            text += "Ground Truth Answer: " + case.answer + "\n"
        if case.retrieved_contexts:
            text += "Retrieved Context: " + "\n".join(case.retrieved_contexts) + "\n"
        if case.generated_answer:
            text += "Generated Answer: " + case.generated_answer + "\n"
        return text
