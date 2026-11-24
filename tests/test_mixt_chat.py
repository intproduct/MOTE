import copy

from fitmotn.chat_formatting import supervision_chat_template
from fitmotn.data.adapters import normalize_chat_messages
from fitmotn.data.contracts import CanonicalSample
from fitmotn.data.clean_release import _canonicalize_row
from fitmotn.runtime import align_mixt_generation_config
from types import SimpleNamespace


def messages():
    return [{"role": "user", "content": "Compute 1+1"},
            {"role": "assistant", "content": None, "reasoning_content": "Use a tool.",
             "tool_calls": [{"type": "function", "function": {"name": "python", "arguments": {"code": "1+1"}}}]},
            {"role": "tool", "content": "2", "tool_call_id": "call1"},
            {"role": "assistant", "content": "2"}]


def test_native_template_annotation_preserves_rendered_bytes():
    from transformers.utils.chat_template_utils import _compile_jinja_template
    template = '''{%- set ns = namespace(last_query_index=0) %}
{%- set ignored = ns.last_query_index %}
{%- for message in messages %}
{%- if message.role == "user" %}{{- 'USER:' + message.content + '\\n' }}
{%- elif message.role == "assistant" %}
    {{- 'ASSISTANT:' + message.content + '\\n' }}
{%- elif message.role == "tool" %}
    {{- '<tool_response>' + message.content + '</tool_response>\\n' }}
{%- endif %}
{%- endfor %}'''
    data = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "call"},
            {"role": "tool", "content": "result"}, {"role": "assistant", "content": "answer"}]
    annotated = supervision_chat_template(template)
    assert annotated != template
    assert _compile_jinja_template(template).render(messages=data) == _compile_jinja_template(annotated).render(messages=data)
    assert supervision_chat_template(annotated) == annotated


def test_tools_reasoning_and_empty_tool_call_content_survive_data_contracts():
    raw = messages()
    normalized = normalize_chat_messages({"messages": raw})
    assert len(normalized) == 4
    assert normalized[1]["tool_calls"] == raw[1]["tool_calls"]
    assert normalized[1]["reasoning_content"] == "Use a tool."
    assert normalized[2]["tool_call_id"] == "call1"
    row = dict(source_name="test", source_revision="r1", source_split="train", source_row_id="1",
               messages=raw, tools=[{"type": "function", "function": {"name": "python"}}])
    canonical = CanonicalSample.from_mapping(row, pipeline_version="test")
    assert canonical.messages[1]["tool_calls"] == raw[1]["tool_calls"]
    assert canonical.metadata["tools"] == row["tools"]
    other = copy.deepcopy(row)
    other["messages"][1]["tool_calls"][0]["function"]["arguments"]["code"] = "2+2"
    assert canonical.canonical_content_hash != CanonicalSample.from_mapping(other, pipeline_version="test").canonical_content_hash
    source = dict(source_name="test", revision="r1", split="train", data_plane="reasoning_sft", adapter="messages",
                  fields={}, license="test", allowed_uses=["training"], acquired_at="2026-11-25")
    a, b = [_canonicalize_row(x, source, 0, "test") for x in (row, other)]
    assert a["messages"][1]["tool_calls"] == row["messages"][1]["tool_calls"]
    assert a["normalized_content_hash"] != b["normalized_content_hash"]
    assert a["tools"] == row["tools"]


def test_native_generation_stops_include_chat_eos_without_dropping_original_stop():
    model = SimpleNamespace(config=SimpleNamespace(mixt_compress_layers=[19], eos_token_id=248044),
                            generation_config=SimpleNamespace(eos_token_id=248044))
    tok = SimpleNamespace(eos_token_id=248046, pad_token_id=248044)
    align_mixt_generation_config(model, tok)
    align_mixt_generation_config(model, tok)
    assert model.config.eos_token_id == [248044, 248046]
    assert model.generation_config.eos_token_id == [248044, 248046]
    assert model.config.pad_token_id == 248044
