# Context Distillation: Important Arguments Reference

## Loss Tau Values (`[trainer.loss]`)

These control the reward composition. The per-sample reward is:
```
reward = adv_tau * task_reward + teacher_tau * sum(log p_teacher) - student_tau * sum(log p_student)
```

| Arg | Default in rl.toml | Description |
|-----|-------------------|-------------|
| `trainer.loss.adv_tau` | `1.0` | Weight for task reward (verifier signal) |
| `trainer.loss.teacher_tau` | `1.0` | Weight for teacher log-prob term (distillation signal) |
| `trainer.loss.student_tau` | `1.0` | Weight for student log-prob term (entropy bonus) |

**Interactions:**
- When `use_full_reward_baseline = true`, these tau values are auto-copied to `orchestrator.advantage.{adv,teacher,student}_tau` by the `RLConfig.auto_setup_advantage_tau` validator. Do not set the orchestrator tau values manually.
- Setting `adv_tau = 0` disables task reward entirely (pure distillation mode).
- Setting `teacher_tau = 0` and `student_tau = 0` disables KL terms (pure RL mode).
- `teacher_tau > 0` requires `teacher_gpu_ids` to be set (validated by `RLConfig.validate_teacher_model`).

## Advantage Configuration (`[orchestrator.advantage]`)

| Arg | Default in rl.toml | Description |
|-----|-------------------|-------------|
| `use_full_reward_baseline` | `true` | Include KL terms in the baseline so group advantages sum to zero |
| `kl_only_incorrect` | `false` | Gate KL terms by `(1 - group_task_reward_mean)` |

### `use_full_reward_baseline`

Controls where KL terms are applied:

- **`true`** (default): KL terms are included in the advantage computation at the **sequence level** in `advantage.py`. The full reward is `adv_tau * task_reward + teacher_tau * sum(teacher_lp) - student_tau * sum(inference_lp)`. The baseline is the group mean of this full reward, so advantages sum to zero. In `loss.py`, the advantage is used as-is (no additional tau scaling or per-token KL).
- **`false`**: Advantages are computed from task reward only. KL terms are added **per-token** in `loss.py`: `advantages += teacher_tau * teacher_lp - student_tau * trainer_lp`. The baseline only subtracts the mean task reward, so the KL component does NOT cancel out across the group.

**Interactions:**
- Synced to `trainer.loss.use_full_reward_baseline` automatically. This flag tells `loss.py` whether to skip its own tau scaling and per-token KL addition.
- When `true`, tau values are copied from `trainer.loss` to `orchestrator.advantage`.

### `kl_only_incorrect`

Gates KL regularization so it is only active when the group lacks task reward signal.

The gate per sample is: `kl_gate = 1 - mean(task_rewards in group)`

| Group outcome | group mean | kl_gate | Effect |
|---|---|---|---|
| All incorrect | 0.0 | 1.0 | KL fully active |
| All correct | 1.0 | 0.0 | KL zeroed out |
| Mixed (e.g. 1/4 correct) | 0.25 | 0.75 | KL 75% active |

**Interactions:**
- With `use_full_reward_baseline = true`: gating is applied in `advantage.py` at the sequence level. The `kl_gates` tensor is also shipped through the pipeline but unused in `loss.py` (the `pass` branch skips it).
- With `use_full_reward_baseline = false`: gating is applied in `loss.py` at the per-token level via the `kl_gates` tensor.
- Only meaningful when `teacher_tau > 0` or `student_tau > 0` (otherwise there are no KL terms to gate).

## GPU Layout (CLI args)

| Arg | Description |
|-----|-------------|
| `--inference_gpu_ids` | GPU IDs for student inference (vLLM) |
| `--teacher_gpu_ids` | GPU IDs for teacher inference (vLLM) |
| `--trainer_gpu_ids` | GPU IDs for training (FSDP2/torchrun) |
| `--inference.parallel.dp N` | Data parallelism for inference (must match number of inference GPUs) |

**Interactions:**
- Teacher and inference can share a GPU (e.g., `--inference_gpu_ids 0,1 --teacher_gpu_ids 1`).
- `--inference.parallel.dp` must equal `len(inference_gpu_ids) / tp`. Auto-inferred by `RLConfig.auto_setup_dp` if inference config exists.
- Number of trainer GPUs determines `orchestrator.num_train_workers` (auto-set by `RLConfig.auto_setup_num_train_workers`).

## Orchestrator Settings (`[orchestrator]`)

| Arg | Default in rl.toml | Description |
|-----|-------------------|-------------|
| `batch_size` | `256` | Total samples per training step |
| `rollouts_per_example` | `16` | Rollouts per prompt (for GRPO grouping) |

**Interactions:**
- `batch_size` must be divisible by `rollouts_per_example`.
- `rollouts_per_example` determines group size for advantage normalization and `kl_only_incorrect` gating.
- Larger `rollouts_per_example` gives better advantage estimates but fewer unique prompts per step.

## Sampling (`[orchestrator.sampling]`)

| Arg | Default in rl.toml | Description |
|-----|-------------------|-------------|
| `max_tokens` | `8192` | Max generation tokens per rollout |
| `temperature` | `1.0` | Sampling temperature |

## Teacher Model (`[orchestrator.teacher_model]`)

| Arg | Default in rl.toml | Description |
|-----|-------------------|-------------|
| `context` | (sokoban prompt) | Text prepended to student prompt for teacher inference |
| `eval_baseline` | `true` | Run student vs teacher eval before training starts |

**Interactions:**
- Setting `context` enables context distillation mode: teacher sees a different prompt than student.
- Without `context`, teacher sees the same prompt as student (standard distillation).
- `eval_baseline` requires `[orchestrator.eval]` to be configured.

## Training Hyperparameters (`[trainer]`)

| Arg | Default in rl.toml | Description |
|-----|-------------------|-------------|
| `trainer.optim.lr` | `1e-5` | Learning rate |
| `trainer.model.lora.rank` | `16` | LoRA rank |
| `trainer.model.lora.alpha` | `64` | LoRA alpha (scaling = alpha/rank = 4.0) |
| `trainer.optim.max_norm` | `1.0` | Gradient clipping norm |

## Common Configurations

### Pure task RL (no teacher)
```bash
--trainer.loss.adv_tau 1.0 --trainer.loss.teacher_tau 0 --trainer.loss.student_tau 0
# No --teacher_gpu_ids needed
```

### Hybrid: task reward + KL (default rl.toml)
```bash
--trainer.loss.adv_tau 1.0 --trainer.loss.teacher_tau 1.0 --trainer.loss.student_tau 1.0
--orchestrator.advantage.use-full-reward-baseline
```

### Hybrid with gated KL (KL only when group is incorrect)
```bash
--trainer.loss.adv_tau 1.0 --trainer.loss.teacher_tau 0.01 --trainer.loss.student_tau 0.01
--orchestrator.advantage.use-full-reward-baseline
--orchestrator.advantage.kl-only-incorrect
```

### Pure distillation (no task reward)
```bash
--trainer.loss.adv_tau 0.0 --trainer.loss.teacher_tau 1.0 --trainer.loss.student_tau 1.0
--orchestrator.buffer.skip-verification
```

### Token-level KL (no full reward baseline)
```bash
--trainer.loss.adv_tau 1.0 --trainer.loss.teacher_tau 0.01 --trainer.loss.student_tau 0.01
--no-orchestrator.advantage.use-full-reward-baseline
```
