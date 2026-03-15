# Context Distillation Experiment

Train models using context distillation. The teacher model sees an enhanced prompt with additional context, while the student sees only the base prompt. The student learns to internalize the behavior implied by the teacher's context.

Two distillation modes are supported:
- **Reverse KL** (default): Teacher computes logprobs on student-generated completions. Student is trained to match teacher's distribution via KL penalty in the advantage.
- **Forward KL** (`forward_kl = true`): Teacher generates its own rollouts with context. Student is trained with an SFT loss to maximize likelihood of teacher completions (conditioned on the student prompt), plus standard RL on its own rollouts.

## Launch Training

```bash
uv run rl @ experiments/context_distill/rl.toml \
  --inference_gpu_ids 0,1 \
  --teacher_gpu_ids 2 \
  --trainer_gpu_ids 3 \
  --inference.parallel.dp 2
```

## Configuration

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `trainer.loss.adv_tau` | Weight for task reward | 0.5 |
| `trainer.loss.teacher_tau` | Weight for teacher log prob term (reverse KL) / SFT weight (forward KL) | 0.5 |
| `trainer.loss.student_tau` | Weight for student log prob term (entropy bonus, reverse KL only) | 0.5 |
| `orchestrator.teacher_model.context` | Extra context for teacher prompt | (required) |
| `orchestrator.teacher_model.forward_kl` | Enable forward KL mode (teacher generates, student learns via SFT) | false |
| `orchestrator.teacher_model.share_teacher_weights` | Include teacher RL samples in training (weight sharing) | false |
| `orchestrator.teacher_model.sft_min_reward_gap` | Only apply SFT loss when teacher outperforms student by this gap | None |
| `orchestrator.teacher_model.eval_baseline` | Run baseline eval before training | false |

### Modes

**Hybrid Mode** (recommended): Combine task reward + context distillation
```toml
[trainer.loss]
adv_tau = 0.5        # Weight for task reward
teacher_tau = 0.5    # Weight for teacher log prob term
student_tau = 0.5    # Weight for student log prob term (entropy)

[orchestrator.teacher_model]
context = "Think step by step before answering.\n\n"

[[orchestrator.env]]
id = "sokoban-env"
args = { ... }
```

**Pure Distillation** (reverse KL): Only teacher signal, no task verification
```toml
[trainer.loss]
adv_tau = 0.0
teacher_tau = 1.0
student_tau = 0.5

[orchestrator.buffer]
skip_verification = true

[orchestrator.teacher_model]
context = "..."
```

**Forward KL Mode**: Teacher generates rollouts, student learns via SFT + RL
```toml
[trainer.loss]
adv_tau = 1.0        # Weight for task reward (RL on student rollouts)
teacher_tau = 1.0    # SFT weight on teacher completions
student_tau = 0.0    # Unused in forward KL mode

[orchestrator.teacher_model]
forward_kl = true
share_teacher_weights = true    # Include teacher rollouts in RL training
context = "Think step by step..."
# sft_min_reward_gap = 0.4     # Optional: only SFT when teacher >> student
```

### GPU Allocation

| GPUs | Inference | Teacher | Trainer |
|------|-----------|---------|---------|
| 4 | 0 | 1 | 2,3 |
| 8 | 0,1 | 2,3 | 4,5,6,7 |

Example with 8 GPUs:
```bash
uv run rl @ experiments/context_distill/rl.toml \
  --inference_gpu_ids 0 1 \
  --teacher_gpu_ids 2 3 \
  --trainer_gpu_ids 4 5 6 7 \
  --inference.parallel.tp 2 \
  --teacher_inference.parallel.tp 2
```

## Customizing the Teacher Context

The teacher context is prepended to the student's prompt. Design it to provide the reasoning/behavior you want the student to learn.

**Example: Chain-of-thought reasoning**
```toml
[orchestrator.teacher_model]
context = """Before providing your solution, follow these steps:
1. Analyze the current state carefully
2. Consider all possible actions
3. Evaluate the consequences of each action
4. Choose the optimal path

Now solve this puzzle:

"""
```

**Example: Expert persona**
```toml
[orchestrator.teacher_model]
context = """You are a world-class puzzle solver with decades of experience.
You never make mistakes and always find the optimal solution.
Approach this methodically:

"""
```

## Baseline Evaluation

Before training starts, you can run a baseline evaluation to compare student vs teacher performance:

```toml
[orchestrator.teacher_model]
context = "Think step by step..."
eval_baseline = true  # Enable baseline eval
```

This logs:
- `eval/{env}/baseline_student/avg` - Student performance (no context)
- `eval/{env}/baseline_teacher/avg` - Teacher performance (with context)
- `eval/{env}/baseline_diff` - The gap the student needs to close

**Note**: Requires `[orchestrator.eval]` to be configured for the baseline eval to run.

## Monitoring

Training metrics are logged to W&B:
- `teacher_kl`: KL divergence from teacher to student (lower = more similar) [reverse KL]
- `combined_advantage`: Combined advantage (task reward + teacher signal) [reverse KL]
- `loss/policy`: Policy gradient loss
- `reward/mean`: Mean task reward (if using hybrid mode)
- `forward_kl/teacher_reward`: Mean teacher rollout reward [forward KL]
- `forward_kl/student_reward`: Mean student rollout reward [forward KL]
- `forward_kl/forward_kl`: Forward KL divergence (teacher || student) [forward KL]
- `forward_kl/sft_gated_groups`: Number of groups passing SFT gate [forward KL, if gating enabled]
- `sft_loss`: SFT loss on teacher completions [forward KL]

## Troubleshooting

### Out of GPU Memory
- Reduce `batch_size` in the config
- Use LoRA: already enabled by default (`trainer.model.lora.rank = 16`)
- Increase tensor parallelism for inference

### Slow Training
- Increase `batch_size` for better GPU utilization
- Check if teacher inference is bottlenecked (reduce `max_tokens`)

### Context Not Working
- Ensure the context ends with appropriate whitespace/newlines
- Check that `prompt_text` is being populated (inspect trajectory data)

## Files

- `rl.toml` - Main training configuration (reverse KL)
- `forward_kl.toml` - Forward KL distillation configuration
- `forward_kl_seed2.toml` - Forward KL with different seed (reproducibility)
- `CLAUDE.md` - Implementation notes for AI assistants
- `README.md` - This file

## See Also

- [On-Policy Distillation Docs](../../docs/on_policy_distillation.md)
- [Sokoban Experiment](../sokoban/README.md)
