# Context Distillation Experiment Log

## Setup

- **Model**: Qwen/Qwen3-4B-Instruct-2507
- **Task**: Sokoban puzzle solving
- **Method**: Reverse-KL on-policy context distillation (sequence-level via `use_full_reward_baseline=true`)
- **Base config**: `experiments/context_distill/rl.toml`
- **GPU split**: 2 inference + 2 teacher + 4 trainer (8 GPUs total)

## Completed Runs

### 16K Equal Weights (pure KL, no task reward)
- **Config**: `rl_equal_weights_16k.toml`
- **W&B**: `sokoban-cd-equal-16k`
- **Params**: adv_tau=0.0, teacher_tau=0.5, student_tau=0.5
- **Steps**: 500, seq_len=16884, rollouts_per_example=8
- **Status**: Completed

### 16K Low Entropy (pure KL, asymmetric weights)
- **Config**: `rl_low_entropy_16k.toml`
- **W&B**: `sokoban-cd-low-entropy-16k`
- **Params**: adv_tau=0.0, teacher_tau=0.5, student_tau=0.05
- **Steps**: 500, seq_len=16884, rollouts_per_example=8
- **Status**: Completed

## New Runs: Sequence-Level KL Context Distillation

### Group A: Task reward ON (adv_tau=1.0), varying KL weight

| Run | Config | adv_tau | teacher_tau | student_tau | W&B Name | Status |
|-----|--------|---------|-------------|-------------|----------|--------|
| A1 | `rl_task1.0_kl0.0.toml` | 1.0 | 0.0 | 0.0 | `cd-task1.0-kl0.0` | Pending |
| A2 | `rl_task1.0_kl0.01.toml` | 1.0 | 0.01 | 0.01 | `cd-task1.0-kl0.01` | Pending |
| A3 | `rl_task1.0_kl0.001.toml` | 1.0 | 0.001 | 0.001 | `cd-task1.0-kl0.001` | Pending |

### Group B: Task reward OFF (adv_tau=0.0), KL only

| Run | Config | adv_tau | teacher_tau | student_tau | W&B Name | Status |
|-----|--------|---------|-------------|-------------|----------|--------|
| B1 | `rl_task0.0_kl1.0.toml` | 0.0 | 1.0 | 1.0 | `cd-task0.0-kl1.0` | Pending |

### Shared Parameters (all new runs)
- max_steps=400, seq_len=16884, max_tokens=8192
- rollouts_per_example=16, batch_size=256
- use_full_reward_baseline=true (sequence-level KL)
- LoRA: rank=16, alpha=64
- LR: 1e-5, betas=(0.85, 0.9)

### Launch Order
A1 → A2 → A3 → B1 (sequential, all 8 GPUs per run)

## Results Summary

| Run | Final Task Reward | Final KL | Final Loss | Notes |
|-----|------------------|----------|------------|-------|
| A1 (task only) | | | | |
| A2 (task + kl0.01) | | | | |
| A3 (task + kl0.001) | | | | |
| B1 (kl only) | | | | |

## Key Questions
1. Does adding KL regularization to task reward improve generalization?
2. What KL weight gives the best task reward / KL tradeoff?
3. How does pure KL (B1, equal weights) compare to task reward + KL (A2, A3)?
