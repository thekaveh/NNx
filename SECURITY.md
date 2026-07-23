# Security Policy

## 1. Supported Versions

Security fixes are applied to the latest release and the current `main` branch. Older releases may not receive patches; upgrade to the newest published version before reporting a version-specific problem.

## 2. Reporting a Vulnerability

Please use [GitHub private vulnerability reporting](https://github.com/thekaveh/NNx/security/advisories/new) rather than opening a public issue. Include the affected version or commit, reproduction steps, impact, and any suggested mitigation. You can expect an acknowledgement within seven days and a status update after the report has been triaged.

Do not include secrets, personal data, or exploit details in public issues, discussions, or pull requests while a report is being investigated.

## 3. Checkpoint Trust Boundary

NNx pickle checkpoints use `torch.load(..., weights_only=False)` to reconstruct Python dataclasses. Loading an untrusted pickle checkpoint can execute arbitrary code. Use safetensors for artifacts from untrusted sources, and only load `.pt` checkpoints produced by a trusted party.

Optimizer and training-state sidecars use `weights_only=True`, but they must still accompany a trusted NNx checkpoint and pass the generation-stamp validation performed by `NNCheckpoint.load_training_state()`.
