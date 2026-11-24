"""Check native tokenizer IDs and supervision after the zero-mask correction."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from transformers import AutoTokenizer
from fitmotn.chat_formatting import make_standard_chat_supervised_example

checkpoint = Path(sys.argv[1])
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
result = {"token_ids": {token: tokenizer.convert_tokens_to_ids(token) for token in
                       ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]},
          "config_eos": json.loads((checkpoint / "config.json").read_text())["eos_token_id"],
          "generation_eos": json.loads((checkpoint / "generation_config.json").read_text())["eos_token_id"],
          "tokenizer_eos": tokenizer.eos_token_id, "cases": {}}
cases = {
    "cot": [{"role": "user", "content": "1+1=?"},
            {"role": "assistant", "content": "<think>Add one and one.</think>2"}],
    "tir": [{"role": "user", "content": "1+1=?"},
            {"role": "assistant", "content": "<tool_call>print(1+1)</tool_call>"},
            {"role": "tool", "content": "2"}, {"role": "assistant", "content": "2"}],
}
for key, messages in cases.items():
    try:
        encoded = make_standard_chat_supervised_example(tokenizer, messages, max_len=1024, enable_thinking=True)
        labels = encoded["labels"]
        result["cases"][key] = {"supervised_tokens": int((labels != -100).sum()),
                                  "supervised_text": tokenizer.decode(labels[labels != -100])}
    except Exception as e:
        result["cases"][key] = {"error": f"{type(e).__name__}: {e}"}
serialized = json.dumps(result, ensure_ascii=False, indent=2)
Path(sys.argv[2]).write_text(serialized, encoding="utf-8")
print(serialized)
