import json
import re
from typing import cast

import jieba
from rouge import Rouge

from rage.results import PrecisionRecallF1Result


def is_valid_json(json_str: str) -> bool:
    try:
        json.loads(json_str, strict=False)
    except json.JSONDecodeError:
        return False
    return True


def extract_json(json_str: str) -> str:
    # markdown code pattern
    json_code_pattern = r"```json\n(.*?)```"
    match = re.search(json_code_pattern, json_str, re.DOTALL)
    if match and is_valid_json(match.group(1)):
        return match.group(1)

    code_pattern = r"```(.*?)```"
    match = re.search(code_pattern, json_str, re.DOTALL)
    if match and is_valid_json(match.group(1)):
        return match.group(1)

    inline_code_pattern = r"`(.*?)`"
    match = re.search(inline_code_pattern, json_str, re.DOTALL)
    if match and is_valid_json(match.group(1)):
        return match.group(1)

    raise ValueError(f"Invalid JSON, output should be a valid JSON string. Got: {json_str}")


def ensure_valid_json(json_str: str) -> str:
    if is_valid_json(json_str):
        return json_str
    return extract_json(json_str)


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
    scores = cast(dict[str, float], scores)
    return PrecisionRecallF1Result(precision=scores["p"], recall=scores["r"], f1=scores["f"])
