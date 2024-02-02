from rage.case import RageCase
from rage.metrics.base import RageMetric
from rage.results import FaithfulnessResult
from rage.utils import caculate_rouge_l_score, calculate_word_overlap, split_chinese_sentences


class WorldOverlapAnswerFaithfulness(RageMetric[FaithfulnessResult]):
    required_fields = {"answer", "retrieved_contexts"}

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def calculate(self, case: RageCase) -> FaithfulnessResult:
        case = self.refine_case(case)
        context = "\n".join(case.retrieved_contexts)
        answer_sentences = split_chinese_sentences(case.answer)
        if not answer_sentences:
            return FaithfulnessResult(faithfulness=0)
        faithful_sentences = []
        non_faithful_sentences = []
        scores = []
        for sentence in answer_sentences:
            word_overlap_p = calculate_word_overlap(sentence, context).precision
            scores.append(word_overlap_p)
            if word_overlap_p > self.threshold:
                faithful_sentences.append(sentence)
            else:
                non_faithful_sentences.append(sentence)
        faithfulness = len(faithful_sentences) / len(answer_sentences)
        return FaithfulnessResult(
            faithfulness=faithfulness,
            extra={
                "faithful_sentences": faithful_sentences,
                "non_faithful_sentences": non_faithful_sentences,
                "answer_sentence_with_word_overlap_score": list(zip(answer_sentences, scores)),
            },
        )


class RougeLAnswerFaithfulness(RageMetric[FaithfulnessResult]):
    required_fields = {"answer", "retrieved_contexts"}

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def calculate(self, case: RageCase) -> FaithfulnessResult:
        case = self.refine_case(case)
        context = "\n".join(case.retrieved_contexts)
        answer_sentences = split_chinese_sentences(case.answer)
        if not answer_sentences:
            return FaithfulnessResult(faithfulness=0)
        faithful_sentences = []
        non_faithful_sentences = []
        scores = []
        for sentence in answer_sentences:
            rouge_l_score = caculate_rouge_l_score(sentence, context).precision
            scores.append(rouge_l_score)
            if rouge_l_score > self.threshold:
                faithful_sentences.append(sentence)
            else:
                non_faithful_sentences.append(sentence)
        faithfulness = len(faithful_sentences) / len(answer_sentences)
        return FaithfulnessResult(
            faithfulness=faithfulness,
            extra={
                "faithful_sentences": faithful_sentences,
                "non_faithful_sentences": non_faithful_sentences,
                "answer_sentence_with_rouge_l_score": list(zip(answer_sentences, scores)),
            },
        )
