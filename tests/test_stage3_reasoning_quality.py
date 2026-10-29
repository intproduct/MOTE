from __future__ import annotations

import pytest

from MOTE.data.sft_format import FORMAT_VERSION, parse_math_raw_v2_target, render_math_raw_v2
from MOTE.tasks.answer_extraction import (
    AnswerValidationError,
    extract_final_answer,
    infer_answer_type,
    validate_final_answer,
)
from MOTE.tasks.reasoning_normalization import ReasoningNormalizationError, normalize_reasoning_sample, select_openr1_trace
from MOTE.data.supervision import SupervisionError
from MOTE.data.tokenization import make_supervised_example


class WordTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(range(len(str(text).split())))


def test_openr1_parallel_arrays_choose_verified_not_short_unverified():
    row = {
        "generations": ["Short. #### 1", "A longer verified derivation with several steps. #### 2"],
        "correctness_math_verify": [False, True],
        "correctness_llama": [True, False],
    }
    selected = select_openr1_trace(row, tokenizer=WordTokenizer())
    assert selected["selected_generation_index"] == 1
    assert selected["selected_math_verified"] is True
    assert selected["selected_llama_verified"] is False
    assert selected["available_generation_count"] == 2
    assert selected["available_verified_count"] == 1


def test_openr1_verified_candidates_use_real_tokenizer_length_and_stable_index():
    row = {
        "generations": ["many words in this verified trace #### 3", "short verified #### 3"],
        "correctness_math_verify": [True, True],
    }
    selected = select_openr1_trace(row, tokenizer=WordTokenizer())
    assert selected["selected_generation_index"] == 1
    assert selected["trace_selection_strategy"] == "math_verified_format_complete_shortest_tokenized"


@pytest.mark.parametrize(
    "row,reason",
    [
        ({"generations": ["#### 1", "#### 2"], "correctness_math_verify": [True]}, "parallel_array_length_mismatch"),
        ({"generations": ["#### 1"], "correctness_math_verify": [False]}, "no_verified_candidate"),
    ],
)
def test_openr1_invalid_verification_is_fail_closed(row, reason):
    with pytest.raises(ReasoningNormalizationError) as exc:
        select_openr1_trace(row, tokenizer=WordTokenizer())
    assert exc.value.reason_code == reason


def test_openr1_dynamic_normalization_refuses_approximate_length_selection():
    result = normalize_reasoning_sample(
        {
            "problem": "q",
            "generations": ["verified #### 1"],
            "correctness_math_verify": [True],
        },
        "openr1_math",
        limits={},
    )
    assert result["ok"] is False
    assert result["reason"] == "production_tokenizer_required"


@pytest.mark.parametrize(
    "text,expected",
    [
        (r"work\nThe answer is: \sqrt{5}", r"\sqrt{5}"),
        (r"work\n#### -12.5", "-12.5"),
        (r"first \boxed{1}; final \boxed{\frac{13}{6}}", r"\frac{13}{6}"),
        (r"Final Answer: \begin{pmatrix}1&2\\3&4\end{pmatrix}", r"\begin{pmatrix}1&2\\3&4\end{pmatrix}"),
        (r"Therefore, the answer is 12 meters", "12 meters"),
    ],
)
def test_final_answer_parser_supports_required_structures(text, expected):
    assert extract_final_answer(text) == expected


def test_final_answer_parser_has_no_unconditional_last_line_fallback():
    assert extract_final_answer("reasoning line\nthis is merely another reasoning line") == ""


def test_answer_type_and_quality_gates():
    assert infer_answer_type(r"\{1,2,3\}") == "set"
    assert infer_answer_type("[0, 1)") == "interval"
    assert infer_answer_type("12 meters") == "unit"
    with pytest.raises(AnswerValidationError) as exc:
        validate_final_answer("proof text", "proof text", task_type="numeric")
    assert exc.value.reason_code == "answer_equals_solution"
    proof = validate_final_answer("proof text", "proof text", task_type="proof")
    assert proof["answer_type"] == "proof"


def test_math_raw_v2_render_parse_round_trip():
    rendered = render_math_raw_v2("What is 2+2?", "2+2=4.", "4")
    assert rendered["format_version"] == FORMAT_VERSION
    assert rendered["prompt"] == "Question:\nWhat is 2+2?"
    assert rendered["target"].endswith("#### 4")
    parsed = parse_math_raw_v2_target(rendered["target"])
    assert parsed == {"reasoning": "2+2=4.", "final_answer": "4"}


@pytest.mark.parametrize("total_tokens", [1279, 1280, 1281])
def test_seq1280_exact_token_boundaries_have_no_truncation(total_tokens):
    tokenizer = WordTokenizer()
    prompt = "p " * 9 + "p"
    target_word_count = total_tokens - 10 - 1
    target = "a " * max(0, target_word_count - 1) + "a"
    if total_tokens <= 1280:
        example = make_supervised_example(tokenizer, prompt, target, max_len=1280, add_eos=True)
        assert example["data_diagnostics"]["total_tokens"] == total_tokens
        assert example["data_diagnostics"]["overflow_tokens"] == 0
        assert example["labels"][:10].tolist() == [-100] * 10
        assert example["input_ids"][-1].item() == tokenizer.eos_token_id
    else:
        with pytest.raises(SupervisionError) as exc:
            make_supervised_example(tokenizer, prompt, target, max_len=1280, add_eos=True)
        assert exc.value.reason_code == "overlong"
        assert exc.value.diagnostics.total_tokens == 1281
