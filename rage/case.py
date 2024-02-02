from pydantic import BaseModel


class RageCase(BaseModel):
    question: str = ""
    contexts: list[str] = []
    answer: str = ""
    retrieved_contexts: list[str] = []
    generated_answer: str = ""


class RageExample[T: BaseModel](BaseModel):
    rage_case: RageCase
    output: T
