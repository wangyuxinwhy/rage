from functools import partial
from typing import Callable, Literal, Self

from rouge import Rouge

from rage.case import RageCase
from rage.metrics.base import RageMetric
from rage.results import PrecisionRecallF1Result
from rage.utils import split_chinese_sentences

type MatchFunction = Callable[[str, str], bool]


def exact_match(retrieved_text: str, ground_truth_text: str) -> bool:
    return retrieved_text == ground_truth_text


def rouge_match(retrieved_text: str, ground_truth_text: str, threshold: float = 0.8) -> bool:
    rouge = Rouge()
    scores = rouge.get_scores(retrieved_text, ground_truth_text)[0]
    return scores["rouge-l"]["r"] >= threshold  # type: ignore


class ContextPrecisionRecallF1(RageMetric[PrecisionRecallF1Result]):
    def __init__(self, split_to_sentence: bool = False, match_function: MatchFunction = exact_match) -> None:
        self.split_to_sentence = split_to_sentence
        self.match_function = match_function

    @classmethod
    def from_parameters(
        cls,
        split_to_sentence: bool = False,
        match_strategy: Literal["exact", "rouge"] = "exact",
        rouge_threshold: float = 0.8,
    ) -> Self:
        if match_strategy == "exact":
            match_function = exact_match
        elif match_strategy == "rouge":
            match_function = partial(rouge_match, threshold=rouge_threshold)
        else:
            raise ValueError(f"Invalid match_strategy: {match_strategy}")
        return cls(split_to_sentence=split_to_sentence, match_function=match_function)

    def calculate(self, case: RageCase) -> PrecisionRecallF1Result:
        assert case.retrieved_contexts is not None
        assert case.contexts is not None
        if self.split_to_sentence:
            retrieved_contexts = [
                sentence for context in case.retrieved_contexts for sentence in split_chinese_sentences(context)
            ]
            contexts = [sentence for context in case.contexts for sentence in split_chinese_sentences(context)]
        else:
            retrieved_contexts = case.retrieved_contexts
            contexts = case.contexts

        num_match = 0
        match_contexts = set()
        for retrieved_context in retrieved_contexts:
            for context in contexts:
                if self.match_function(retrieved_context, context):
                    match_contexts.add(context)
                    num_match += 1
                    continue
        precision = num_match / len(retrieved_contexts)
        recall = len(match_contexts) / len(set(contexts))
        if precision == 0 and recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return PrecisionRecallF1Result(
            precision=precision,
            recall=recall,
            f1=f1,
        )
