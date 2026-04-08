from src.generation.prompt_builder import build_baseline_prompt
from src.generation.llm import generate_answer


def answer_without_rag(query: str) -> str:
    prompt = build_baseline_prompt(query)
    return generate_answer(prompt)