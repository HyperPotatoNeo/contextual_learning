# Contextual Learning Repository Guide

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Set environment variables
export WANDB_API_KEY=<your-wandb-key>
unset NCCL_SOCKET_IFNAME  # Required for distributed training
```

## Architecture Overview

This is an async RL framework with 4 main components:

1. **Inference Server** (`inference`) - vLLM-based, generates rollouts
2. **Orchestrator** (`orchestrator`) - CPU process, collects trajectories, computes advantages
3. **Trainer** (`trainer`) - Multi-GPU FSDP2 training
4. **Master** (`rl`) - Orchestrates all components together

## Entrypoints

| Command | Purpose |
|---------|---------|
| `rl @ config.toml` | Full pipeline (inference + orchestrator + trainer) |
| `trainer @ config.toml` | RL trainer only |
| `sft @ config.toml` | SFT trainer |
| `inference @ config.toml` | Inference server only |
| `orchestrator @ config.toml` | Orchestrator only |
| `eval @ config.toml` | Evaluation |

## Configuration System

Uses TOML files with `@` syntax. Precedence (highest to lowest):
1. CLI args: `--key.subkey value`
2. TOML files: `@ path/to/config.toml`
3. Environment: `PRIME_KEY__SUBKEY=value`
4. Defaults

### Key Config Sections

```toml
max_steps = 100              # Training steps
seq_len = 2048               # Max sequence length
max_async_level = 2          # Off-policy horizon
output_dir = "outputs"

[model]
name = "Qwen/Qwen3-0.6B"     # HuggingFace model ID

[trainer.optim]
lr = 1e-5

[trainer.model.lora]         # Optional LoRA
rank = 32
alpha = 64

[orchestrator]
batch_size = 128
rollouts_per_example = 8

[[orchestrator.env]]
id = "reverse-text"          # Environment name

[wandb]
project = "my-project"
```

## Common Commands

### Debug/Validation Tests
```bash
# Test RL trainer (fake data, 5 steps)
trainer @ configs/debug/rl/train.toml

# Test SFT trainer
sft @ configs/debug/sft/train.toml
```

### Single-GPU Training
```bash
# Full RL pipeline on single GPU
rl @ examples/reverse_text/rl.toml --trainer_gpu_ids 0 --inference_gpu_ids 1

# SFT only
sft @ examples/reverse_text/sft.toml
```

### Multi-GPU Training
```bash
# Using torchrun for trainer
torchrun --nproc-per-node 4 src/prime_rl/trainer/rl/train.py @ config.toml
```

## Context Distillation

This fork adds **context distillation** - training a student to match a teacher that has additional context:

```bash
# Run context distillation experiment
rl @ experiments/context_distill/rl.toml \
  --inference_gpu_ids 0 \
  --teacher_gpu_ids 1 \
  --trainer_gpu_ids 2,3
```

See `experiments/context_distill/README.md` for details.

## Examples

| Example | Task | Model | Notes |
|---------|------|-------|-------|
| `examples/reverse_text/` | Reverse strings | 0.6B | Entry level, SFT + RL |
| `examples/alphabet_sort/` | Sort names | 4B + LoRA | No SFT warmup needed |
| `examples/wordle/` | Play Wordle | 1.7B | Multi-turn |
| `examples/wiki_search/` | Trivia w/ tools | - | Tool use |
| `experiments/context_distill/` | Context distillation | Any | Teacher-student setup |

## Directory Structure

```
contextual_learning/
├── src/prime_rl/
│   ├── trainer/rl/train.py      # RL trainer entry
│   ├── trainer/sft/train.py     # SFT trainer entry
│   ├── inference/server.py      # vLLM inference
│   ├── orchestrator/            # Data orchestration
│   └── rl.py                    # Master orchestrator
├── configs/
│   ├── debug/                   # Minimal test configs
│   └── ...                      # Domain configs
├── examples/                    # End-to-end examples
└── experiments/
    └── context_distill/         # Context distillation experiment
```

## Typical Workflow

1. **SFT** (optional warmup):
   ```bash
   sft @ examples/reverse_text/sft.toml --output-dir outputs/sft
   ```

2. **RL Training**:
   ```bash
   rl @ examples/reverse_text/rl.toml --model.name outputs/sft/weights/step_100
   ```

3. **Evaluation**:
   ```bash
   eval @ configs/debug/eval.toml --model.name outputs/rl/weights/step_20
   ```

## Important Notes

- Always `unset NCCL_SOCKET_IFNAME` before running distributed training
- Use `export WANDB_MODE=disabled` to skip W&B logging during tests
- Checkpoints saved to `output_dir/weights/step_N`
- Debug configs in `configs/debug/` use fake data and run in seconds
- The `@` syntax loads TOML configs: `command @ config.toml`

## Loss Functions

Available via `trainer.loss.type`:
- `aipo` (default) - Advantage-weighted IPO
- `grpo` - Group Relative Policy Optimization
- `gspo` - Group Squared Policy Optimization
- `opo` - Online Policy Optimization
- `rloo` - REINFORCE Leave-One-Out

### Context Distillation Loss Parameters

```toml
[trainer.loss]
adv_tau = 0.5       # Weight for task reward
teacher_tau = 0.5   # Weight for teacher log prob (distillation)
student_tau = 0.5   # Weight for student log prob (entropy bonus)
```

## LoRA Training

```toml
[trainer.model.lora]
rank = 32
alpha = 64
target_modules = ["q_proj", "v_proj"]  # Optional
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| NCCL socket error | `unset NCCL_SOCKET_IFNAME` |
| OOM | Reduce `batch_size` or `seq_len` |
| Model not found | Check HF cache or use full path |
| W&B errors | `export WANDB_MODE=disabled` |
