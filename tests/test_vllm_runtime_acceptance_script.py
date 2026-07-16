from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_vllm_runtime_acceptance.py"
SPEC = importlib.util.spec_from_file_location("validate_vllm_runtime_acceptance", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    def convert_ids_to_tokens(self, token_id):
        return f"piece-{int(token_id)}"

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "|".join(str(int(token)) for token in token_ids)


def test_runtime_diagnostics_extract_prompt_and_ranked_logprobs():
    output = SimpleNamespace(
        prompt_token_ids=[1, 2, 3],
        outputs=[
            SimpleNamespace(
                token_ids=[7],
                logprobs=[
                    {
                        7: SimpleNamespace(logprob=-0.1, rank=1, decoded_token="seven"),
                        8: SimpleNamespace(logprob=-1.2, rank=2, decoded_token="eight"),
                    }
                ],
            )
        ],
    )

    assert MODULE._observed_prompt_token_ids([output]) == [[1, 2, 3]]
    rows = MODULE._vllm_first_position_logprobs([output], FakeTokenizer())
    assert [row["token_id"] for row in rows] == [7, 8]
    assert rows[0]["rank"] == 1
    assert rows[0]["token_piece"] == "piece-7"
    assert rows[0]["decoded"] == "7"


def test_hf_token_rank_uses_strictly_greater_scores():
    logprobs = torch.tensor([-3.0, -1.0, -2.0, -1.0])
    assert MODULE._hf_token_rank(logprobs, 1) == 1
    assert MODULE._hf_token_rank(logprobs, 2) == 3
