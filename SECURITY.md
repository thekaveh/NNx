# 20. Security Policy

## 1. Supported Versions

Security fixes are applied to the latest release and the current `main` branch. Older releases may not receive patches; upgrade to the newest published version before reporting a version-specific problem.

## 2. Reporting a Vulnerability

Please use [GitHub private vulnerability reporting](https://github.com/thekaveh/NNx/security/advisories/new) rather than opening a public issue. Include the affected version or commit, reproduction steps, impact, and any suggested mitigation. You can expect an acknowledgement within seven days and a status update after the report has been triaged.

Do not include secrets, personal data, or exploit details in public issues, discussions, or pull requests while a report is being investigated.

## 3. Checkpoint Trust Boundary

NNx pickle checkpoints use `torch.load(..., weights_only=False)` to reconstruct Python dataclasses. Loading an untrusted pickle checkpoint can execute arbitrary code. Use safetensors for artifacts from untrusted sources, and only load `.pt` checkpoints produced by a trusted party.

Optimizer and training-state sidecars use `weights_only=True`, but they must still accompany a trusted NNx checkpoint and pass the generation-stamp validation performed by `NNCheckpoint.load_training_state()`.

To share a run, export it as a **run bundle** (`nnx.bundles.export_bundle`): export reads your own pickle checkpoint and writes its weights, training state and calibrators as safetensors plus schema-validated JSON. The bundle readers (`inspect_bundle`, `validate_bundle`, `reconstruct_bundle`) never unpickle, import code named by the bundle or download anything. They check every payload's SHA-256 and size, the generation id, and that no payload is missing, unlisted, symlinked or outside the bundle before any tensor is read, and a registered module is rebuilt only from factories the caller supplies. A bundle is data, but it is still a model: reconstruction builds the architecture its parameters describe (as `from_pretrained` builds the one `config.json` describes), so read an untrusted bundle's `inspect_bundle(...).model_params` first, and validate where it came from before serving its predictions.
