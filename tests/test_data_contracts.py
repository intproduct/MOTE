from __future__ import annotations

import pytest

from MOTE.data.adapters import clean_text as adapter_clean_text
from MOTE.data.contracts import CanonicalSample, DataContractError, SourceIdentity
from MOTE.data.text_normalization import normalize_text
from MOTE.tasks.answer_extraction import extract_boxed_answer, extract_final_answer


def test_text_normalization_preserves_newlines_latex_and_indentation():
    raw = "\r\nQuestion  \r\n\r\n  code()\t\r\n\\boxed{\\frac{1}{2}}\r\n"
    expected = "Question\n\n  code()\n\\boxed{\\frac{1}{2}}"
    assert normalize_text(raw) == expected
    assert adapter_clean_text(raw) == expected


def test_balanced_boxed_parser_handles_nested_latex_and_keeps_last_answer():
    text = "First \\boxed{1}.\nFinally \\boxed{\\frac{1}{2}}"
    assert extract_boxed_answer(text) == r"\frac{1}{2}"
    assert extract_final_answer(text) == r"\frac{1}{2}"


def test_canonical_identity_is_stable_and_pipeline_version_is_derived_identity():
    row = {
        "source_name": "demo",
        "source_revision": "abc",
        "source_split": "train",
        "source_row_id": "7",
        "prompt": "Q\n  indented",
        "target": "A",
    }
    first = CanonicalSample.from_mapping(row, pipeline_version="v1")
    same = CanonicalSample.from_mapping(dict(row), pipeline_version="v1")
    changed_pipeline = CanonicalSample.from_mapping(row, pipeline_version="v2")
    assert first.identity.source_sample_id == same.identity.source_sample_id
    assert first.derived_sample_id == same.derived_sample_id
    assert changed_pipeline.identity.source_sample_id == first.identity.source_sample_id
    assert changed_pipeline.derived_sample_id != first.derived_sample_id
    assert first.prompt == "Q\n  indented"


def test_raw_and_canonical_hashes_are_distinct_when_transport_noise_is_normalized():
    row = {
        "source_name": "demo",
        "source_revision": "abc",
        "source_split": "train",
        "source_row_id": "8",
        "prompt": "Q  \r\n",
        "target": "A",
    }
    sample = CanonicalSample.from_mapping(row, pipeline_version="v1")
    assert sample.prompt == "Q"
    assert sample.raw_content_hash != sample.canonical_content_hash


def test_source_identity_fails_closed_when_row_identity_is_missing():
    with pytest.raises(DataContractError, match="source identity") as exc:
        SourceIdentity.from_mapping({"source_name": "demo", "source_revision": "abc"})
    assert exc.value.reason_code == "missing_source_identity"
