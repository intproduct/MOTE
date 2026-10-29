# MoTE autoregressive gate comparison

This experiment trains a small causal Transformer **from scratch** on real
WikiText-2 text. It tests next-token prediction, rather than the BERT
classification protocol in `experiments/gate_benchmark`.

## Frozen method

- Data: `Salesforce/wikitext`, `wikitext-2-raw-v1`, immutable revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`. Official train/validation/test
  remain separate. Exact article overlap is checked and rejected. Preserve
  upstream raw text; append EOS between articles. Byte-level BPE (4096 tokens,
  full byte alphabet) is fitted **only on train**. Data/tokenizer hashes and
  source parquet hashes are recorded. Public source and license information:
  https://huggingface.co/datasets/Salesforce/wikitext .
- Model: 2 pre-LN decoder blocks, hidden 256, 4 attention heads, FFN 1024,
  learned positions, context 128, tied input/output embeddings, no dropout,
  FP32. Causal SDPA attention. Both FFN projections use the existing MoTE
  expert implementation, 16 experts, top-4, no token dropping. Shared
  embedding/attention/norm/expert weights and expert positions are byte-identical
  across routers within each seed. This is a small architecture study, not a
  pretrained LLM or a state-of-the-art language model.
- Six routers: full/half MLP, depth-2 MPO with chi=2/full or chi=1/half,
  depth-1 amplitude Born full/half. Same definitions as the completed second
  classification round. Budget includes LN, input projection, circuit/MPO,
  output projection and learned gain; integer widths produce approximate
  budgets. The parameter reduction refers to routers, not the whole model.
  The MPO is classical/nonunitary. Born uses real joint amplitudes, RY/CRY
  evolution, and squared-amplitude readout. Its classical input rank is not
  an MPS bond dimension. No hardware quantum speedup is claimed.
- Next-token labels are shifted once. Train windows are random contiguous
  chunks sampled with replacement; equal seeds share independent minibatch
  generators and recorded window indices. AdamW, cosine learning rate,
  warmup, clip norm 1, auxiliary routing loss 0.01. The repository's existing
  hard-top-k plus soft straight-through gradient rule is retained unchanged.
- Validation selects lowest **token CE**, not aux-inclusive training loss.
  PPL=exp(total NLL / scored tokens). Evaluation uses context 128 and stride
  64, with past targets masked and every token after the initial EOS scored
  exactly once. Right padding is excluded from routing and loss. Overlapping
  context tokens are counted repeatedly in *routing* statistics, explicitly
  labeled. This BPE PPL is not comparable to word-level published WikiText PPL.
  Method reference: https://huggingface.co/docs/transformers/perplexity .
- The pilot uses only a validation prefix and never reads test tokens.
  Formal training uses full validation, and test is evaluated only after the
  entire matrix completes. No test-based architecture/checkpoint selection.
- Training compute seconds exclude evaluation/checkpoint writes; wall time
  includes them. A separate serial 5-warmup/20-step AdamW benchmark starts
  from each best checkpoint, uses fresh optimizer state and disposable weights.
  Include CE/PPL curves against both steps/tokens and measured train seconds.

## Execution

Use the existing TITAN V Python environment and single GPU worker. Set
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `PYTHONUTF8=1`. Avoid concurrent GPU
training because the previous 6-worker classification run caused a Windows
driver watchdog reboot. Do not stop unrelated processes.

```powershell
.venv-titan/python.exe -m fitmotn.experiments.autoregressive_gate.runner prepare --config experiments/autoregressive_gate/configs/pilot.json
.venv-titan/python.exe -m fitmotn.experiments.autoregressive_gate.runner run --config experiments/autoregressive_gate/configs/pilot.json
```

Pilot acceptance checks numerical finiteness, all router tensors updating,
shared initialization and minibatch order, reload equality, split integrity,
and complete artifacts. Gate ranking is not an acceptance criterion. Select
the formal compute budget from measured pilot throughput and memory before
opening the formal test split; use three seeds (17/29/43).

The selected formal configuration is `configs/titan_v.json`: batch 32,
4000 steps, warmup 200, full-validation selection every 500 steps, seeds
17/29/43. Each run processes 16,384,000 sampled training tokens (about 5.31
times the training-corpus token count; sampling is with replacement). This
was chosen from two successful six-router resource/stability pilots at batch
8 and 32, not test scores. Run the serial background pipeline with:

```powershell
.venv-titan/python.exe -u -m fitmotn.experiments.autoregressive_gate.pipeline --config experiments/autoregressive_gate/configs/titan_v.json
```

The pipeline runs all 18 trainings, audit, best-checkpoint test reevaluation,
generation, serial timing, and CSV/JSON/figure generation. `ready_for_review`
means measured explanations and visual review remain; it does not declare
the research task complete. The reviewed report is protected from accidental
regeneration by the analysis script.

Each configuration has its own hash directory under
`outputs/autoregressive_gate/{pilot,formal}/`. Completed runs are verified and
skipped. Incomplete directories are never overwritten or silently resumed;
keep an interruption archive before an explicit same-seed retry. `last.pt`
contains weights and metrics, **not** optimizer/RNG resume state. A failed run
stops the pipeline and retains its log, checkpoint and traceback. Models,
data and training artifacts stay ignored by Git.

## Scientific checks

`tests/test_autoregressive_gate.py` checks future-token causality, batch/padding
independence, exact label shifts, strided target coverage including tails,
token-weighted PPL, shared state/budgets, actual router updates and checkpoint
restore. Existing independent MPO/Born correctness tests remain applicable.

Report all seeds and paired differences in CE and PPL relative to full/half
MLP. Distinguish equal-token capability from elapsed-time efficiency. Include
actual router and total parameters, throughput, memory, best checkpoint step,
entropy/load CV/unused experts, and fixed-prompt generation samples without
cherry-picking. Three seeds and a small single corpus cannot establish broad
LLM, quantum or entanglement advantage.
