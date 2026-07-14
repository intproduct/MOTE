# Extra Datasets

FitMoTN supports a generic `data.extra_datasets` list for standard local JSONL and HuggingFace-style datasets without adding dataset-specific config fields.

Each item requires `name`, `source`, and `format`.

- `format`: `text`, `chat_messages`, `prompt_response`, or `reasoning_qa`
- `source`: `hf`, `local_jsonl`, `jsonl`, `jsonl_gz`, `load_from_disk`, or `auto`
- `group`: defaults to `pretrain` for `text`, otherwise `task`
- `bucket`: defaults to `pretrain_general` for pretrain tasks, otherwise `extra_task`

Inspect before training:

```bash
python -m fitmotn.cli.inspect_dataset --config path/to/config.json --max-samples 3
python -m fitmotn.cli.inspect_dataset --config path/to/config.json --tokenizer /path/to/tokenizer
```

Tulu or AM-style messages:

```json
{
  "name": "tulu_olmoe_sft",
  "source": "hf",
  "hf_name": "allenai/tulu-v3.1-mix-preview-4096-OLMoE",
  "split": "train",
  "format": "chat_messages",
  "messages_field": "messages",
  "weight": 0.6,
  "max_samples": 610000,
  "group": "task",
  "bucket": "general_sft",
  "source_family": "chat"
}
```

Infinity or ShareGPT-style conversations:

```json
{
  "name": "infinity_gen",
  "source": "hf",
  "hf_name": "BAAI/Infinity-Instruct",
  "hf_config": "Gen",
  "split": "train",
  "format": "chat_messages",
  "messages_field": "conversations",
  "role_key": "from",
  "content_key": "value",
  "role_map": {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system"
  },
  "group": "task",
  "bucket": "general_sft"
}
```

Alpaca-style prompt/response:

```json
{
  "name": "local_alpaca",
  "source": "local_jsonl",
  "path": "data/alpaca.jsonl",
  "format": "prompt_response",
  "instruction_field": "instruction",
  "input_field": "input",
  "output_field": "output",
  "group": "task"
}
```

Plain text:

```json
{
  "name": "local_text",
  "source": "local_jsonl",
  "path": "data/text.jsonl",
  "format": "text",
  "text_field": "text",
  "group": "pretrain"
}
```
