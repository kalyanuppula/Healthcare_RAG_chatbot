import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag_chain import get_rag_answer

def test_diabetes_question():
    result = get_rag_answer("What factors increase the risk of developing type 2 diabetes?")

    assert result["answer"]
    assert "diabetes.txt" in result["sources"]


def test_malaria_question():
    result = get_rag_answer("What are the symptoms of malaria?")

    assert "could not find relevant information" in result["answer"].lower()
    assert result["sources"] == []

def test_hypertension_question():
    result = get_rag_answer("What is hypertension?")

    assert result["answer"]
    assert "hypertension.txt" in result["sources"]

def test_diabetes_answer_mentions_insulin():
    result = get_rag_answer("What is diabetes?")

    assert "diabetes.txt" in result["sources"]
    assert "insulin" in result["answer"].lower()

def test_unrelated_question():
    result = get_rag_answer("Who won the FIFA World Cup in 2022?")

    assert "could not find relevant information" in result["answer"].lower()
    assert result["sources"] == []

def test_empty_question():
    result = get_rag_answer("")

    assert result["answer"] == "Please enter a healthcare question."
    assert result["sources"] == []