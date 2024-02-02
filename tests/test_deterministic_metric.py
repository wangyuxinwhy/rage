from rage.case import RageCase
from rage.metrics import (
    ContextPrecisionRecallF1,
    RougeLAnswerCorrectness,
    RougeLAnswerFaithfulness,
    WorldOverlapAnswerCorrectness,
    WorldOverlapAnswerFaithfulness,
)


def test_context_precision_recall_f1():
    case = RageCase(
        retrieved_contexts=[
            "Paris is the capital of France and also the largest city in the country.",
            "Lyon is a major city in France.",
        ],
        contexts=["Paris is the capital of France."],
    )
    metric = ContextPrecisionRecallF1()
    result = metric.calculate(case)
    assert isinstance(result.precision, float)
    assert isinstance(result.recall, float)
    assert isinstance(result.f1, float)


def test_world_overlap_answer_correctness():
    case = RageCase(
        answer="Shakespeare wrote 'Romeo and Juliet'",
        generated_answer="Shakespeare",
    )
    metric = WorldOverlapAnswerCorrectness()
    result = metric.calculate(case)
    assert isinstance(result.correctness, float)


def test_rouge_l_answer_correctness():
    case = RageCase(
        answer="Shakespeare wrote 'Romeo and Juliet'",
        generated_answer="Shakespeare",
    )
    metric = RougeLAnswerCorrectness()
    result = metric.calculate(case)
    assert isinstance(result.correctness, float)


def test_world_overlap_answer_faithfulness():
    case = RageCase(
        answer="Paris is the capital of France.",
        retrieved_contexts=["Paris is the capital of France."],
    )
    metric = WorldOverlapAnswerFaithfulness()
    result = metric.calculate(case)
    assert isinstance(result.faithfulness, float)


def test_rouge_l_answer_faithfulness():
    case = RageCase(
        answer="Paris is the capital of France.",
        retrieved_contexts=["Paris is the capital of France."],
    )
    metric = RougeLAnswerFaithfulness()
    result = metric.calculate(case)
    assert isinstance(result.faithfulness, float)
