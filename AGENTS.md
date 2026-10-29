# Workspace and delivery conventions

- The active QuantumGate experiment workspace is `D:/files/MOTE`, on branch `mote-quantum`. Check the actual repository path, branch, and dirty state before editing or publishing. Do not infer the correct branch from an old chat's working directory.
- `D:/files/pydocs/MOTE` is a separate checkout containing local dataset assets. Keep those assets in place. For QuantumGate experiments, use the active workspace explicitly.
- Publish current MoTE/QuantumGate work to `mote-quantum` unless the user explicitly selects another branch. Do not roll back or force-push `mote-sft-v2` merely because it contains historical misplaced commits.
- When delivering a human-facing Markdown report, also provide a corresponding Typst source. If a local Typst compiler is available, compile and verify a PDF. Machine-facing instruction files such as this file do not require a print companion.
- Preserve frozen experiment source/configuration and original output artifacts. Historical helpers archived under `research/workspace_reconciliation_20261029/archived_helpers` are evidence, not automatically safe experiment launchers; review their original paths before running them.
