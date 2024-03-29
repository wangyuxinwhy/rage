from typing import Any, Dict, Generic, List, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class RageCase(BaseModel):
    question: str = ""
    contexts: List[str] = []
    answer: str = ""
    retrieved_contexts: List[str] = []
    generated_answer: str = ""
    extra: Dict[str, Any] = {}

class RageExample(BaseModel, Generic[T]):
    rage_case: RageCase
    output: T
