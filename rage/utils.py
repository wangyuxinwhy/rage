from __future__ import annotations

import re
from typing import Dict, cast

import jieba
from rouge import Rouge

from rage.results import PrecisionRecallF1Result


def split_chinese_sentences(text: str) -> list[str]:
    sentence_delimiters = re.compile(r"[。？！!?；]+")
    return [s.strip() for s in sentence_delimiters.split(text) if s]


def calculate_word_overlap(text: str, reference_text: str, split_to_sentence: bool = False) -> PrecisionRecallF1Result:
    if text.strip() == "" or reference_text.strip() == "":
        return PrecisionRecallF1Result(precision=0.0, recall=0.0, f1=0.0)

    if split_to_sentence:
        text_tokens = {word for sentence in split_chinese_sentences(text) for word in jieba.cut(sentence)}
        reference_text_tokens = {word for sentence in split_chinese_sentences(reference_text) for word in jieba.cut(sentence)}
    else:
        text_tokens = set(jieba.cut(text))
        reference_text_tokens = set(jieba.cut(reference_text))

    num_overlap = len(text_tokens & reference_text_tokens)
    if num_overlap == 0:
        return PrecisionRecallF1Result(precision=0.0, recall=0.0, f1=0.0)

    precision = num_overlap / len(text_tokens)
    recall = num_overlap / len(reference_text_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return PrecisionRecallF1Result(precision=precision, recall=recall, f1=f1)


def caculate_rouge_l_score(text: str, reference_text: str) -> PrecisionRecallF1Result:
    rouge = Rouge()
    if text.strip() == "" or reference_text.strip() == "":
        return PrecisionRecallF1Result(precision=0.0, recall=0.0, f1=0.0)

    scores = rouge.get_scores(text, reference_text)[0]["rouge-l"]
    scores = cast(Dict[str, float], scores)
    return PrecisionRecallF1Result(precision=scores["p"], recall=scores["r"], f1=scores["f"])
