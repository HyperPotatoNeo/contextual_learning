# Context Distillation Implementation Notes

This file documents the reverse-KL on-policy context distillation feature for future reference.

## What is Context Distillation?

Context distillation trains a student model to match a teacher model's behavior, where the teacher has access to additional context (e.g., system prompt with instructions) that the student doesn't see. The student learns to internalize the behavior implied by the teacher's context.

**Key insight**: Same model weights, different prompts. The teacher sees an enhanced prompt, the student sees the base prompt.

## Architecture

```
Student Inference (GPU 0):
  [User: Base Prompt] → vLLM → [Completion] + student_logprobs

Teacher Inference (GPU 1):
  [User: Context + Base Prompt] + [Completion] → vLLM prefill → teacher_logprobs

Reward:
  reward = sum_t (teacher_logprob[t] - student_logprob[t])

Loss:
  advantage = adv_tau * task_reward + teacher_tau * log(p_teacher) - student_tau * log(p_student)
```

The teacher context is prepended to the user prompt and then the chat template is applied, ensuring proper formatting for instruction-tuned models.

## Implementation Details

### Files Modified

1. **`src/prime_rl/transport/types.py`**
   - Added `prompt_text: str | None = None` to `TrainingSample`
   - This carries the original prompt text for context distillation

2. **`src/prime_rl/orchestrator/trajectories.py`**
   - Modified `interleave_rollout()` and `branch_rollout()`
   - Now populates `prompt_text=first_step.get("prompt")` from the trajectory state

3. **`src/prime_rl/orchestrator/config.py`**
   - Added `context: str | None = None` to `TeacherModelConfig`
   - Added `eval_baseline: bool = False` to `TeacherModelConfig`
   - When context is set, enables context distillation mode
   - When eval_baseline is True, runs baseline eval before training

4. **`src/prime_rl/orchestrator/utils.py`**
   - Added `compute_teacher_logprobs_with_context()` function
   - Prepends teacher context to user prompt, then applies chat template
   - Builds messages: `[{"role": "user", "content": context + prompt}]`
   - Uses `tokenizer.apply_chat_template()` for proper formatting
   - Returns only completion logprobs (aligned with student sequence)

5. **`src/prime_rl/orchestrator/orchestrator.py`**
   - Added import for `compute_teacher_logprobs_with_context`
   - Added import for `run_context_distillation_baseline_eval`
   - Modified teacher logprobs section to check `config.teacher_model.context`
   - If context is set: use new function + pad with zeros for prompt positions
   - If context is None: use original function (backward compatible)
   - Added baseline eval call before training loop if `eval_baseline=True`

6. **`src/prime_rl/eval/utils.py`**
   - Added `run_context_distillation_baseline_eval()` function
   - Runs eval twice: student baseline (no context) and teacher baseline (with context)
   - Logs metrics as `eval/{env}/baseline_student/avg` and `eval/{env}/baseline_teacher/avg`

### Key Code Paths

**Context distillation enabled** (`config.teacher_model.context` is set):
```python
# orchestrator.py
teacher_logprobs_list = await compute_teacher_logprobs_with_context(
    clients=teacher_clients,
    model_name=teacher_model_name,
    samples=train_examples,
    tokenizer=tokenizer,
    teacher_context=config.teacher_model.context,
)
# Pad with zeros for prompt positions
for train_example, completion_logprobs in zip(train_examples, teacher_logprobs_list):
    prompt_logprobs = [0.0] * len(train_example.prompt_ids)
    train_example.teacher_logprobs = prompt_logprobs + completion_logprobs
```

**Standard distillation** (`config.teacher_model.context` is None):
```python
# orchestrator.py - unchanged behavior
teacher_logprobs_list = await compute_teacher_logprobs(...)
for train_example, teacher_logprobs in zip(train_examples, teacher_logprobs_list):
    train_example.teacher_logprobs = teacher_logprobs
```

### Logprob Alignment

The teacher prompt is longer due to the prepended context. The function handles this by:
1. Prepending teacher context to user prompt: `context + "\n\n" + prompt_text`
2. Building messages and applying chat template: `tokenizer.apply_chat_template(messages, add_generation_prompt=True)`
3. Computing logprobs for `teacher_prompt_ids + completion_ids`
4. Returning only `logprobs[len(teacher_prompt_ids):]` (completion portion)
5. Orchestrator pads with zeros for prompt positions to match student sequence length

### Loss Computation

The loss computation in `src/prime_rl/trainer/rl/loss.py` handles teacher logprobs with separate weights:
```python
# Separate weights for teacher log prob and student log prob (entropy) terms
advantages = loss_config.adv_tau * advantages
if teacher_logprobs is not None:
    advantages = advantages + loss_config.teacher_tau * teacher_logprobs.detach() - loss_config.student_tau * trainer_logprobs.detach()
```

The combined advantage (task reward + teacher signal) is logged as `combined_advantage` in W&B.

## Configuration

### Hybrid Mode (Task Reward + Context Distillation)
```toml
[trainer.loss]
adv_tau = 0.5        # Weight for task reward
teacher_tau = 0.5    # Weight for teacher log prob term (distillation signal)
student_tau = 0.5    # Weight for student log prob term (entropy bonus)

[orchestrator.teacher_model]
context = "Think step by step before answering.\n\n"
```

### Pure Context Distillation (No Task Reward)
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

## Baseline Evaluation

Before training starts, you can optionally run a baseline evaluation that compares:
- **Student baseline**: Model performance without the teacher context
- **Teacher baseline**: Model performance with the teacher context prepended

This helps measure:
1. The effect of the context on model performance
2. The gap the student needs to close during training

### Enable Baseline Eval
```toml
[orchestrator.teacher_model]
context = "Think step by step..."
eval_baseline = true  # Run baseline eval before training
```

### Logged Metrics
- `eval/{env}/baseline_student/avg` - Student avg reward (no context)
- `eval/{env}/baseline_teacher/avg` - Teacher avg reward (with context)
- `eval/{env}/baseline_diff` - Difference (teacher - student)

### Implementation
The baseline eval modifies each example's prompt by prepending the teacher context:
```python
# For string prompts
modified_example["prompt"] = teacher_context + example["prompt"]

# For chat format (list of messages)
# Inserts as system message or prepends to existing system message
```

## GPU Allocation

Typical 4-GPU setup:
- GPU 0: Student inference (vLLM)
- GPU 1: Teacher inference (vLLM, same model)
- GPU 2-3: Trainer (FSDP2)

```bash
uv run rl @ experiments/context_distill/rl.toml \
  --inference_gpu_ids 0 \
  --teacher_gpu_ids 1 \
  --trainer_gpu_ids 2 3
```

## Backward Compatibility

All changes are additive:
- Configs without `[orchestrator.teacher_model].context` use original behavior
- The `prompt_text` field defaults to `None` (ignored when not using context distillation)
- Original `compute_teacher_logprobs` function is NOT modified

## Testing

Run the light tests:
```bash
source .venv/bin/activate
python -c "
from prime_rl.transport.types import TrainingSample
from prime_rl.orchestrator.config import TeacherModelConfig
from prime_rl.orchestrator.utils import compute_teacher_logprobs_with_context

# Test that all imports work
print('All imports successful')
"
```

## Future Improvements

1. **Per-sample context**: Allow different contexts for different samples
2. **Context caching**: Cache tokenized context prefix for efficiency
3. **Metrics**: Add specific context distillation metrics to W&B
