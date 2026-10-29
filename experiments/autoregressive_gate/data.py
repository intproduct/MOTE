from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
import numpy as np
import torch

DATASET_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def articles(lines):
    """WikiText top-level '= Title =' headers delimit articles; retain text verbatim."""
    result, current = [], []
    for text in lines:
        if re.match(r'^\s*= [^=].*[^=] =\s*$', text) and current:
            value = ''.join(current)
            if value.strip():
                result.append(value)
            current = []
        current.append(text)
    if ''.join(current).strip():
        result.append(''.join(current))
    return result


def prepare(root, vocab_size=4096):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root/'manifest.json').exists():
        meta = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        assert meta['vocab_size'] == vocab_size
        for name, sha in meta['files'].items():
            assert digest(root/name) == sha, f'data changed: {name}'
        return meta
    corpora, source = {}, {}
    for split in ('train', 'validation', 'test'):
        name = f'wikitext-2-raw-v1/{split}-00000-of-00001.parquet'
        path = hf_hub_download('Salesforce/wikitext', name, repo_type='dataset', revision=DATASET_REVISION)
        lines = pq.read_table(path)['text'].to_pylist()
        corpora[split] = articles(lines)
        source[split] = {'rows': len(lines), 'articles': len(corpora[split]),
                         'parquet_sha256': digest(path), 'file': name}
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                  special_tokens=['<pad>', '<eos>'],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(corpora['train'], trainer=trainer)
    tokenizer.save(str(root/'tokenizer.json'))
    assert tokenizer.get_vocab_size() == vocab_size
    eos = tokenizer.token_to_id('<eos>')
    sets = {k: {hashlib.sha256(t.encode()).hexdigest() for t in v} for k, v in corpora.items()}
    overlap = {f'{a}/{b}': len(sets[a]&sets[b]) for a, b in
               [('train','validation'), ('train','test'), ('validation','test')]}
    if any(overlap.values()):
        raise ValueError(f'exact article overlap across official splits: {overlap}')
    for split, texts in corpora.items():
        tokens = [eos]
        for text in texts:
            tokens.extend(tokenizer.encode(text).ids)
            tokens.append(eos)
        np.save(root/f'{split}.npy', np.asarray(tokens, dtype=np.int32))
        source[split]['tokens'] = len(tokens)
        source[split]['text_sha256'] = hashlib.sha256(''.join(texts).encode()).hexdigest()
        source[split]['article_sha256'] = sorted(sets[split])
    meta = {'dataset': 'Salesforce/wikitext', 'subset': 'wikitext-2-raw-v1',
            'revision': DATASET_REVISION, 'tokenizer_training_split': 'train',
            'tokenizer': 'byte-level BPE, no text normalization, EOS between articles',
            'vocab_size': vocab_size, 'pad_id': 0, 'eos_id': eos,
            'exact_article_overlap': overlap, 'splits': source,
            'files': {p.name: digest(p) for p in sorted(root.glob('*.npy'))}}
    meta['files']['tokenizer.json'] = digest(root/'tokenizer.json')
    write_json(root/'manifest.json', meta)
    return meta


def training_batch(tokens, batch_size, context, generator, device):
    # Random contiguous windows, with replacement. The generator is router independent.
    starts = torch.randint(len(tokens)-context, (batch_size,), generator=generator)
    block = tokens[starts[:, None] + torch.arange(context+1)[None, :]]
    return block[:, :-1].to(device), block[:, 1:].to(device), starts


def evaluation_batches(tokens, context, stride, batch_size, device):
    """Strided evaluation: each token 1..N-1 is scored exactly once.

    The first window scores context targets. Subsequent windows mask the
    already scored prefix. Right padding is excluded from routing and loss.
    """
    if not 1 <= stride <= context:
        raise ValueError('stride must be in [1, context]')
    pending = []
    target_start = 1
    while target_start < len(tokens):
        count = context if target_start == 1 else stride
        target_end = min(len(tokens), target_start+count)
        begin = max(0, target_end-1-context)
        x = tokens[begin:target_end-1]
        y = tokens[begin+1:target_end].clone()
        y[:target_start-begin-1] = -100
        pending.append((x, y))
        target_start = target_end
        if len(pending) == batch_size or target_start == len(tokens):
            ids = torch.zeros(len(pending), context, dtype=torch.long)
            labels = torch.full_like(ids, -100)
            valid = torch.zeros_like(ids, dtype=torch.bool)
            for i, (x, y) in enumerate(pending):
                ids[i, :len(x)], labels[i, :len(y)], valid[i, :len(x)] = x, y, True
            yield ids.to(device), labels.to(device), valid.to(device)
            pending.clear()
