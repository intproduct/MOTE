from __future__ import annotations

from typing import Any

from ..tasks.answer_extraction import clean_text, extract_hash_answer, normalize_math_answer


FORMAT_VERSION = "math_raw_v2"


def render_math_raw_v2(problem: Any, reasoning: Any, final_answer: Any) -> dict[str, str]:
    problem_text = clean_text(problem)
    reasoning_text = clean_text(reasoning)
    answer_text = normalize_math_answer(clean_text(final_answer))
    if not problem_text or not reasoning_text or not answer_text:
        raise ValueError("math_raw_v2 requires non-empty problem, reasoning, and final_answer")
    return {
        "format_version": FORMAT_VERSION,
        "prompt": f"Question:\n{problem_text}",
        "target": f"Solution:\n{reasoning_text}\n\n#### {answer_text}",
    }


def parse_math_raw_v2_target(target: Any) -> dict[str, str]:
    text = clean_text(target)
    if not text.startswith("Solution:\n"):
        raise ValueError("math_raw_v2 target must start with 'Solution:'")
    marker = "\n\n#### "
    if marker not in text:
        raise ValueError("math_raw_v2 target is missing the final-answer marker")
    reasoning, answer_segment = text[len("Solution:\n") :].rsplit(marker, 1)
    answer = extract_hash_answer(f"#### {answer_segment}")
    if not clean_text(reasoning) or not answer:
        raise ValueError("math_raw_v2 target has empty reasoning or final answer")
    return {"reasoning": clean_text(reasoning), "final_answer": normalize_math_answer(answer)}
