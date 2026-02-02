# Sokoban Experiments

Sokoban puzzle-solving environment using [reasoning-gym](https://github.com/open-thought/reasoning-gym).

## Environment

The sokoban-env generates Sokoban puzzles where the model must solve grid-based box-pushing puzzles.

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_train_examples` | 1000 | Training set size |
| `num_eval_examples` | 100 | Eval set size |
| `seed` | 42 | Random seed |
| `min_w` / `max_w` | 4 / 5 | Grid width range |
| `min_h` / `max_h` | 4 / 5 | Grid height range |
| `min_boxes` / `max_boxes` | 1 / 3 | Number of boxes |
| `max_depth` | 80 | Max solution depth |

## Setup

Install the environment:
```bash
prime env install hyperpotatoneo/sokoban-env
```

Verify:
```bash
python -c "from verifiers import load_environment; env = load_environment('sokoban-env'); print(env)"
```

## Training (RL)

### Quick Start

```bash
uv run rl @ experiments/sokoban/rl.toml
```

### GPU Configuration

By default: GPU 0 = inference, GPU 1 = trainer. Override with CLI flags:

```bash
# Use GPU 0 for inference, GPUs 1-3 for training
uv run rl @ experiments/sokoban/rl.toml \
  --inference-gpu-ids 0 \
  --trainer-gpu-ids 1,2,3

# Single GPU (time-share inference and training)
uv run rl @ experiments/sokoban/rl.toml \
  --inference-gpu-ids 0 \
  --trainer-gpu-ids 0
```

### Config Reference (`rl.toml`)

```toml
max_steps = 200                    # Training steps
seq_len = 4096                     # Max sequence length
output_dir = "outputs/sokoban"

[model]
name = "Qwen/Qwen3-4B-Instruct-2507"

[ckpt]
interval = 50                      # Save every N steps
keep_last = 3                      # Keep last N checkpoints

[wandb]
project = "sokoban-rl"
name = "sokoban-4b-lora"

[trainer.model.lora]               # LoRA settings
rank = 32
alpha = 64

[trainer.optim]
lr = 1e-5                          # Learning rate
weight_decay = 0.01
max_norm = 1.0                     # Gradient clipping

[orchestrator]
batch_size = 256                   # Global batch size
rollouts_per_example = 8           # Rollouts per prompt

[orchestrator.sampling]
max_tokens = 4096                  # Max generation tokens
temperature = 1.0                  # Sampling temperature

[[orchestrator.env]]
id = "sokoban-env"
args = { num_train_examples = 5000, min_boxes = 3, max_boxes = 8 }
```

### Key Parameters

| Section | Parameter | Description |
|---------|-----------|-------------|
| Top-level | `max_steps` | Total training steps |
| Top-level | `seq_len` | Max context length |
| `[trainer.model.lora]` | `rank`, `alpha` | LoRA rank and scaling |
| `[trainer.optim]` | `lr` | Learning rate |
| `[orchestrator]` | `batch_size` | Global batch size |
| `[orchestrator]` | `rollouts_per_example` | Rollouts per prompt |
| `[orchestrator.sampling]` | `temperature` | Sampling temperature |
| `[orchestrator.sampling]` | `max_tokens` | Max generation length |

### Output

Checkpoints saved to: `outputs/sokoban/weights/step_N/`

## Evaluation

### Baseline (before training)

Start the inference server:
```bash
inference --model.name Qwen/Qwen3-4B-Instruct-2507
```

Run eval:
```bash
.venv/bin/eval @ experiments/sokoban/eval.toml --client.base-url http://localhost:8000/v1
```

### After Training

Eval the trained checkpoint:
```bash
inference --model.name outputs/sokoban/weights/step_200

.venv/bin/eval @ experiments/sokoban/eval.toml \
  --client.base-url http://localhost:8000/v1 \
  --model.name outputs/sokoban/weights/step_200
```

### Using vf-eval CLI

```bash
vf-eval sokoban-env \
  -m Qwen/Qwen3-4B-Instruct-2507 \
  -b http://localhost:8000/v1 \
  -n 100 \
  --max-tokens 4096 \
  --temperature 1.0
```

With custom env args:
```bash
vf-eval sokoban-env \
  -m Qwen/Qwen3-4B-Instruct-2507 \
  -b http://localhost:8000/v1 \
  -n 50 \
  --max-tokens 4096 \
  --env-args '{"min_boxes": 3, "max_boxes": 8}'
```

## Eval Config Reference (`eval.toml`)

```toml
num_examples = 200
rollouts_per_example = 1

[model]
name = "Qwen/Qwen3-4B-Instruct-2507"

[sampling]
max_tokens = 4096
temperature = 1.0

[[env]]
id = "sokoban-env"
args = { min_w = 4, max_w = 8, min_h = 4, max_h = 8, min_boxes = 3, max_boxes = 8 }
```
