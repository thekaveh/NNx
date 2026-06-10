# DPO — Direct Preference Optimization

`nnx.dpo_train_step_factory` implements the Direct Preference
Optimization objective from Rafailov et al. (2023) — a simpler
alternative to PPO-based RLHF that fits in a single supervised-style
training loop. Combined with `nnx.NNPreferenceDataset` and an
`nnx.GenerativeNNModel` (the decoder-only LM path; see [`docs/lm.md`](lm.md)), DPO becomes a
drop-in `train_step_fn=` for the standard `NNModel.train(...)` call.

## 1. What DPO does

Given a frozen **reference policy** `π_ref` (typically the SFT
checkpoint the model was warm-started from) and a trainable **policy**
`π_θ`, DPO directly fits the policy to a dataset of
`(prompt, chosen_response, rejected_response)` triples by minimising:

```
L_DPO = -log σ(β · ( (log π_θ(y_w | x) - log π_ref(y_w | x))
                   - (log π_θ(y_l | x) - log π_ref(y_l | x)) ))
```

— i.e., maximise the policy's log-ratio margin (chosen over rejected)
relative to the reference's. There's no separate reward model and no
RL loop; the standard `NNModel.train()` machinery just runs.

## 2. When DPO beats SFT

SFT (supervised fine-tuning on `(prompt, good_response)` pairs)
maximises the likelihood of "good" responses but says nothing about
what's *worse*. When you have explicit preference data —
`A > B` for the same prompt — DPO almost always outperforms SFT on the
target preference metric, because it directly optimises the *gap*
between chosen and rejected.

Pick DPO over SFT when:

- You have preference pairs already (a published HF Hub dataset like
  Anthropic HH, OpenAssistant, or UltraFeedback; or annotator-labelled
  data from your own pipeline).
- The SFT model is already producing fluent output and you want to
  steer *style* / *behaviour* (helpfulness, harmlessness, refusal
  format, brevity, etc.).
- You don't have the infrastructure for a full PPO RLHF loop with a
  separate reward model.

Pick SFT (or layered SFT → DPO) over DPO when:

- You have lots of "good" examples but very few labelled
  comparison pairs.
- The base model is still learning the target task's basic format —
  preference data won't fix raw fluency.

## 3. Quickstart

```python
import torch
from nnx import (
    Devices, GenerativeNNModel, Losses, Nets, NNModelParams,
    NNOptimParams, NNPreferenceDataset, NNSchedulerParams,
    NNTokenizerParams, NNTrainParams, NNTransformerParams, Optims,
    dpo_train_step_factory, set_seed, train_bpe,
)

set_seed(0)

# 1. Tokenizer (BPE for the demo; swap in a published tokenizer for
#    real preference data).
tk = train_bpe(files=None, texts=["..."], vocab_size=8192,
               special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"])
tokenizer = NNTokenizerParams.of(tokenizer=tk, path="artifacts/tok.json")

# 2. Build the policy (the trainable model) and load SFT weights.
net_params = NNTransformerParams(
    input_dim=tokenizer.vocab_size,
    output_dim=tokenizer.vocab_size,
    dropout_prob=0.0,
    vocab_size=tokenizer.vocab_size,
    n_layers=4, n_heads=4, d_model=128,
    ffn_mult=4, max_seq_len=128,
)
model_params = NNModelParams(net=Nets.TRANSFORMER, device=Devices.CPU,
                             loss=Losses.CROSS_ENTROPY)
policy = GenerativeNNModel(net_params=net_params, params=model_params,
                           tokenizer=tokenizer)
# policy.net.load_state_dict(torch.load("sft-checkpoint.pt"))   # in practice

# 3. Build the reference (a frozen copy of the SFT model).
ref_model = GenerativeNNModel(net_params=net_params, params=model_params,
                              tokenizer=tokenizer)
ref_model.net.load_state_dict(policy.net.state_dict())

# 4. Preference dataset — yields (prompt_ids, chosen_ids, rejected_ids).
preferences = NNPreferenceDataset(
    prompts=["..."],
    chosen=["..."],
    rejected=["..."],
    tokenizer=tokenizer,
    max_prompt_len=64,
    max_response_len=64,
    pad_token_id=1,  # "<pad>"
    batch_sizes=(8, 8, 8),
    seed=0,
)

# 5. DPO step — frozen reference + β temperature.
# pad_token_id matches the dataset's — padded response positions are
# excluded from the log-prob sums.
step_fn = dpo_train_step_factory(ref_model, beta=0.1, pad_token_id=1)

# 6. Train. NNModel.train() machinery is unchanged — callbacks, schedulers,
#    checkpointing all work as usual.
policy.train(
    params=NNTrainParams(
        n_epochs=5,
        train_loader=preferences.train_loader,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=5e-5,
                            momentum=(0.9, 0.999), weight_decay=0.0),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5,
                                    patience=2, cooldown=1, threshold=1e-3),
    ),
    train_step_fn=step_fn,
)
```

## 4. The `beta` knob

`beta` controls how sharply the policy is allowed to diverge from the
reference. The original paper recommends `0.1` as the default; values
in `[0.01, 0.5]` are common in practice.

- **Higher β** (e.g. 0.5): the implicit reward is steeper. The policy
  can drift further from the reference per gradient step. Risk: the
  policy stops being a *language model* and starts gaming the
  preference function.
- **Lower β** (e.g. 0.01): the policy stays close to the reference.
  Slower convergence but safer.

If training diverges or the model collapses (output goes to gibberish
or to a single fixed answer), lower `beta` first.

## 5. Pair with HF Hub preference datasets

`NNPreferenceDataset` takes three parallel lists of strings — easy to
fill from the `datasets` library:

```python
from datasets import load_dataset

ds = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")
preferences = NNPreferenceDataset(
    prompts=[row["chosen"].split("Assistant:", 1)[0] for row in ds],
    chosen=[row["chosen"].split("Assistant:", 1)[1] for row in ds],
    rejected=[row["rejected"].split("Assistant:", 1)[1] for row in ds],
    tokenizer=tokenizer,
    max_prompt_len=256,
    max_response_len=256,
    pad_token_id=1,
)
```

The convention `(prompt, chosen, rejected)` matches the published
datasets directly — most HF Hub preference corpora ship in this shape
or one isomorphic to it.

## 6. Honest scope

NNx's DPO is built for **small-LM experimentation** — the same
TinyStories-class sub-30-minute-on-a-laptop scope as the LM path's
`GenerativeNNModel`. It is **not** a production RLHF replacement.
Specifically:

- The training step does two forward passes through the policy and
  two through the reference, per row. For a `~7B` parameter model on
  a single GPU this dominates memory and time; production stacks
  cache reference log-probs offline or share the reference with the
  policy via LoRA.
- There's no IPO / cDPO / RPO variant, no offline reference log-prob
  cache, no PEFT integration (LoRA-DPO is the obvious next step but
  isn't wired in for v1).
- Mixed precision and gradient accumulation are explicitly **not**
  supported by the DPO step (the
  [paradigm finalize_step](https://github.com/thekaveh/NNx/blob/main/src/nnx/_step_helpers.py)
  contract raises rather than silently dropping these knobs).
- No multi-GPU / multi-node sharding.

For production-scale preference tuning, use a dedicated stack
(`trl`, `axolotl`, `OpenRLHF`, etc.) and treat NNx's DPO as the
"how does this objective behave on my small LM?" experimentation
path. The implementation is intentionally short and readable — the
whole training step is ~30 lines — so you can also use it as a
reference for understanding the DPO loss before scaling up.

## 7. How it composes with the rest of NNx

- **`train_step_fn=` hook** — `dpo_train_step_factory` returns a
  standard `TrainStepFn`; same plug shape as KD, SimCLR, Mixup,
  CutMix, MoE, and the diffusion paradigm.
- **`NNRun` content-addressed persistence** — DPO runs hash the same
  way as any other run. The reference model isn't part of the run
  config (it's an opaque dependency); track its provenance separately
  (file hash, HF Hub revision, etc.).
- **Callbacks** — `EarlyStopping`, `ModelCheckpoint`,
  `TensorBoardCallback`, `WandbCallback` all work unchanged on the
  policy. The DPO step reports `loss` (the DPO loss) and `error`
  (negated chosen−rejected log-prob gap, so lower is better — same
  monotone direction as `loss`).
- **Generation** — after training, `policy.generate(prompt=..., ...)`
  produces text from the tuned policy with the standard sampling
  knobs (top-k, top-p, repetition penalty, seed).
