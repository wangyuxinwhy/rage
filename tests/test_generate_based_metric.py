from rage.case import RageCase
from rage.metrics import (
    GenerateBasedAnswerCorrectness,
    GenerateBasedAnswerFaithfulness,
    GenerateBasedAnswerRelevance,
    GenerateBasedContextCoverage,
    GenerateBasedContextPrecision,
)


def test_context_coverage():
    case = RageCase(
        question="What is the largest and second city in France?",
        retrieved_contexts=[
            "Lyon is a major city in France.",
            "Paris is the capital of France and also the largest city in the country.",
        ],
        answer="Paris is the largest city in France and Marseille is the second largest.",
    )
    metric = GenerateBasedContextCoverage.from_parameters(model_id="openai/gpt-3.5-turbo")
    result = metric.calculate(case)
    assert isinstance(result.coverage, float)


def test_context_precision():
    case = RageCase(
        question="What is the capital of France?",
        retrieved_contexts=[
            "Paris is the capital of France and also the largest city in the country.",
            "Lyon is a major city in France.",
        ],
    )
    metric = GenerateBasedContextPrecision.from_parameters(model_id="openai/gpt-3.5-turbo")
    result = metric.calculate(case)
    assert isinstance(result.precision, float)


def test_correctness():
    case = RageCase(
        question="Who wrote 'Romeo and Juliet'?",
        answer="Shakespeare wrote 'Romeo and Juliet'",
        generated_answer="Shakespeare",
    )
    metric = GenerateBasedAnswerCorrectness.from_parameters(model_id="openai/gpt-3.5-turbo")
    result = metric.calculate(case)
    assert isinstance(result.correctness, float)


def test_faithfulness():
    case = RageCase(
        question="What is the capital of France?",
        retrieved_contexts=["Paris is the capital of France."],
        generated_answer="Paris",
    )
    metric = GenerateBasedAnswerFaithfulness.from_parameters(model_id="openai/gpt-3.5-turbo")
    result = metric.calculate(case)
    assert isinstance(result.faithfulness, float)


def test_relevance():
    case = RageCase(
        question="Who wrote 'Romeo and Juliet'?",
        answer="Shakespeare wrote 'Romeo and Juliet'",
    )
    metric = GenerateBasedAnswerRelevance.from_parameters(model_id="openai/gpt-3.5-turbo")
    result = metric.calculate(case)
    assert isinstance(result.relevance, float)
