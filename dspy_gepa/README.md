# GEPA Prompt Optimization for Sokoban

This project uses [GEPA (Genetic-Pareto)](https://dspy.ai/) prompt optimization from DSPy to optimize system prompts for solving Sokoban puzzles with Qwen3-4B, targeting [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) context distillation.

## Quick Start

```bash
# 1. Start vLLM server (in container on compute node, use merged LoRA model)
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/shared_kl_step150_merged \
    --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
    --tensor-parallel-size 4 \
    --max-model-len 16384 \
    --port 8000

# 2. Clear DSPy cache and run GEPA with teacher prompt
rm -rf "$HOME/.dspy_cache" && mkdir -p "$HOME/.dspy_cache"
python -u sokoban_gepa.py --reflection-model claude-opus --teacher-prompt-file teacher_prompt.txt
```

**CRITICAL**: Always clear the DSPy cache before runs. See "Known DSPy Pitfalls" below.

## Results

### Latest: Teacher Prompt + mb=20 (2026-02-17)

Using the full ~95-line teacher prompt as GEPA starting point with reflection minibatch size 20:

| Metric | Value |
|--------|-------|
| Student model | Qwen3-4B (shared_kl_step150 merged LoRA) |
| Reflection model | Claude Opus |
| Baseline (teacher prompt) | 55.5% (1000 test) |
| Best val score | **61.3%** (150 val, Program 5, iter 14) |
| Test score (optimized) | 56.7% (1000 test) |
| Budget | 10000 metric calls, ~2.5h on A100x4 |

**Key findings**:
- mb=20 converges faster than mb=5 (best at iter 14 vs iter 25), fewer false-positive subsamples
- Val-to-test generalization gap: 61.3% val → 56.7% test suggests overfitting to the small 150-example val set
- The winning prompt's key innovation is truncation management — bolded imperatives and explicit "BUDGET YOUR REASONING" guidance
- `teacher_prompt.txt` provides the strong starting point (loaded via `--teacher-prompt-file`)

### Historical: Run 7 (First Correct Run, generic prompt)

| Program | Val Score (200 ex) | Prompt |
|---------|-------------------|--------|
| 0 (baseline) | 37% | Generic reasoning prompt |
| 2 (best) | **38%** | Structured ~40-line prompt |

**Conclusion**: Starting from the teacher prompt (55.5%) is vastly better than the generic prompt (37%). The teacher prompt encodes domain-specific Sokoban knowledge that GEPA cannot easily rediscover from scratch.

## Architecture

```
dspy_gepa/
├── sokoban_gepa.py          # Main GEPA script (PrimeRLAdapter, direct rg.create_dataset)
├── teacher_prompt.txt       # Teacher prompt (~95 lines) used as GEPA starting point
├── merge_lora_weights.py    # Merge LoRA adapters into base model for vLLM serving
├── eval_prompt_standalone.py # Standalone eval (bypasses DSPy, direct API calls)
├── PROGRESS_NOTES.md        # Detailed progress notes with all findings
├── outputs/                 # GEPA output directories (checkpoints, optimized programs)
└── README.md                # This file
```

### Key Components

- **PrimeRLAdapter**: Custom DSPy adapter that sends raw system+user messages for SokobanSolver (matching prime-rl format) while using ChatAdapter for GEPA reflection signatures. Identifies signatures by field names, not `__name__`.

- **Dataset**: Uses `rg.create_dataset("sokoban", ...)` directly. Do NOT use `verifiers.ReasoningGymEnv` — it uses a different index range producing harder puzzles.

- **Metric**: Returns `dspy.Prediction(score=0.0 or 1.0, feedback="...")`. Extracts moves from `<answer>` tags, scores with `sokoban_scorer.score_answer()`.

## Usage

### Running GEPA

```bash
# Always clear DSPy cache first!
rm -rf "$HOME/.dspy_cache"
mkdir -p "$HOME/.dspy_cache"

# Recommended: use teacher prompt as starting point with mb=20
python -u sokoban_gepa.py \
    --reflection-model claude-opus \
    --teacher-prompt-file teacher_prompt.txt \
    --train-size 500 --val-size 150 --test-size 1000 \
    --max-metric-calls 10000 \
    --reflection-minibatch-size 20 \
    --reflection-max-tokens 16384 \
    --num-threads 32 \
    --temperature 1.0 \
    --max-tokens 8192
```

### Resuming from Checkpoint

```bash
python -u sokoban_gepa.py \
    --reflection-model claude-opus \
    --teacher-prompt-file teacher_prompt.txt \
    --resume-dir outputs/gepa_<timestamp> \
    --train-size 500 --val-size 150 --test-size 1000 \
    --max-metric-calls 10000 \
    --reflection-minibatch-size 20 \
    --reflection-max-tokens 16384 \
    --num-threads 32
```

### Standalone Evaluation (No DSPy)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# Use the full GEPA-optimized prompt from PROGRESS_NOTES.md (Program 2)
# or a simple fallback:
SYSTEM_PROMPT = "You are going to solve a 'sokoban' puzzle."

response = client.chat.completions.create(
    model="Qwen/Qwen3-4B-Instruct-2507",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "<sokoban puzzle>"},
    ],
    max_tokens=8192,
    temperature=1.0,
)
```

## Merging LoRA Weights

prime-rl saves RL checkpoints with LoRA adapters separate from base weights. vLLM needs a single merged model directory. Use `merge_lora_weights.py` to merge them.

### Checkpoint structure (prime-rl output)

```
prime-rl/outputs/<run_name>/weights/step_<N>/
├── model-00001-of-00002.safetensors   # Base model weights (frozen during RL)
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── config.json
├── tokenizer.json, tokenizer_config.json, ...
└── lora_adapters/
    ├── adapter_config.json             # LoRA rank, alpha, target_modules
    └── adapter_model.bin               # LoRA A/B weights + full-weight overrides
```

### Usage

```bash
# From inside the container (needs torch + safetensors)
python merge_lora_weights.py <checkpoint_dir> <output_dir>

# Example: merge step_150 from a prime-rl run
python merge_lora_weights.py \
    /pscratch/sd/s/siddart2/prime-rl/outputs/shared_kl_001_task_1_1/weights/step_150 \
    /pscratch/sd/s/siddart2/dspy_gepa/models/shared_kl_step150_merged

# Example: merge from a contextual_learning run
python merge_lora_weights.py \
    /pscratch/sd/s/siddart2/contextual_learning/outputs/<run_name>/weights/step_<N> \
    ./models/<run_name>_step<N>_merged
```

### What it does

1. Loads base model weights from safetensors shards
2. Loads LoRA adapter weights (`lora_A`, `lora_B` pairs)
3. Computes `merged = base + (alpha/rank) * (lora_B @ lora_A)` for each target module
4. Handles `train_lm_head=true`: when `tie_word_embeddings=true` during training, the adapter only saves `embed_tokens` (PyTorch deduplicates tied params). The script syncs `lm_head.weight` from the trained `embed_tokens` and sets `tie_word_embeddings=false` in the output config for inference.
5. Saves merged weights in the same shard layout, copies tokenizer files

### Then serve with vLLM

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./models/shared_kl_step150_merged \
    --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
    --tensor-parallel-size 4 \
    --max-model-len 16384 \
    --port 8000
```

The `--served-model-name` must match the model name used in `sokoban_gepa.py --model` (default: `Qwen/Qwen3-4B-Instruct-2507`).

## Known DSPy Pitfalls

Four critical bugs were discovered and fixed during development. **All fixes are in `sokoban_gepa.py`**, but understanding them prevents future issues:

1. **Adapter must use `dspy.configure(adapter=...)`** — Setting `lm.adapter` does nothing; DSPy reads from global settings only.

2. **Dataset creation path matters** — `verifiers.ReasoningGymEnv` offsets eval indices by `num_train_examples` (5000), producing different (harder) puzzles. Always use `rg.create_dataset()` directly.

3. **DSPy disk cache corrupts cross-run evaluations** — DSPy caches stochastic responses at temperature>0 in `$HOME/.dspy_cache/`. Must clear in shell script BEFORE Python starts (DSPy opens the DB on import).

4. **`with_instructions()` breaks custom adapter routing** — GEPA's `build_program()` calls `signature.with_instructions()` which changes `__name__` from "SokobanSolver" to "StringSignature". Custom adapters must check field names, not `__name__`.

See `PROGRESS_NOTES.md` for full details on all 15 lessons learned.

## API Keys & Environment Variables

Two API keys are needed:

| Variable | Purpose | Where to set |
|----------|---------|--------------|
| `ANTHROPIC_API_KEY` | Claude reflection model (GEPA uses Claude Opus to propose prompt improvements) | Pass via `--anthropic-api-key` flag, or set as env var (DSPy reads `ANTHROPIC_API_KEY` automatically) |
| `WANDB_API_KEY` | Weights & Biases logging (optional) | Set as env var before running the script |

**Example:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export WANDB_API_KEY="..."  # optional

# Then run GEPA
./run_gepa_opus_v7.sh
```

The vLLM student model does not need an API key (`api_key="dummy"` is used for the local server).

## Bugs Fixed to Get GEPA Working

Getting DSPy GEPA to produce correct results required fixing 4 critical bugs across 7 runs. Each bug silently degraded accuracy without raising errors, making them hard to detect.

### Bug 1: DSPy Adapter Not Actually Used (Run 3)

**Symptom**: Set `lm.adapter = PrimeRLAdapter()` but the model still produced short ChatAdapter-style responses with `[[ ## answer ## ]]` markers instead of reasoning.

**Root cause**: DSPy's `Predict.forward()` reads the adapter from `dspy.settings.adapter`, not from `lm.adapter`. Setting it on the LM instance is silently ignored.

**Fix**: Use `dspy.configure(lm=student_lm, adapter=PrimeRLAdapter())` to set the adapter globally.

**Impact**: ChatAdapter format caused Qwen3-4B to skip reasoning entirely (23 tokens avg vs 5000+ tokens with raw format), dropping accuracy from ~37% to ~13%.

### Bug 2: Dataset Index Mismatch (Run 4)

**Symptom**: Standalone eval showed 50% accuracy but GEPA baseline showed 12.5% on the "same" task.

**Root cause**: `verifiers.ReasoningGymEnv` creates a combined dataset of `5000 + N` items and uses indices `5000..5000+N` for evaluation. Direct `rg.create_dataset("sokoban", size=N)` uses indices `0..N`. These produce completely different puzzles — the high-index ones happened to be harder.

**Fix**: Changed `load_datasets()` to use `rg.create_dataset()` directly instead of going through `verifiers.ReasoningGymEnv`.

### Bug 3: DSPy Disk Cache Returning Stale Results (Run 5)

**Symptom**: Baseline accuracy dropped to 12% despite all previous fixes being in place.

**Root cause**: DSPy caches all LM responses in `$HOME/.dspy_cache/` (a SQLite database via `diskcache.FanoutCache`). Even at `temperature > 0`, DSPy caches the first stochastic sample and returns the exact same response for identical requests in subsequent runs. Previous debug scripts had cached bad responses for the same prompts.

**Fix**: Clear the cache in the shell script *before* Python starts:
```bash
rm -rf "$HOME/.dspy_cache"
mkdir -p "$HOME/.dspy_cache"
```
Cannot clear in Python — DSPy opens the database on `import dspy`, so deleting it after import causes "unable to open database file" errors.

### Bug 4: GEPA `with_instructions()` Breaks Adapter Routing (Run 6)

**Symptom**: Baseline test showed 34% (correct) but GEPA's internal evaluations showed 13.5% (wrong). GEPA was optimizing prompts while evaluating them in the wrong format.

**Root cause**: GEPA's `build_program()` calls `signature.with_instructions(new_prompt)` which creates a *new* signature class with `__name__ = "StringSignature"` instead of `"SokobanSolver"`. The PrimeRLAdapter was checking `__name__` to decide raw vs ChatAdapter format, so after `with_instructions()` it silently fell back to ChatAdapter.

**Fix**: Changed `_is_raw_format_signature()` to check field names (`{"question"} -> {"answer"}`) instead of `__name__`, since field structure is preserved through `with_instructions()`.

### Timeline

| Run | Bugs present | Val accuracy | Status |
|-----|-------------|-------------|--------|
| 1-2 | Adapter not used, wrong dataset | 13-20% | Wrong format |
| 3 | Adapter set on LM (ignored) | 12.6% | Bug 1 discovered |
| 4 | Wrong dataset indices | 12.5% | Bug 2 discovered |
| 5 | Stale cache | 12% | Bug 3 discovered |
| 6 | `with_instructions()` breaks adapter | 13.5% val, 34% baseline | Bug 4 discovered |
| 7 | All fixed | **38%** | First correct run |

See `PROGRESS_NOTES.md` for the full investigation log with evidence and 15 lessons learned.

### Extracting the Optimized Prompt

GEPA-optimized prompts can be very long (the Run 8 best prompt is 102 lines). Grepping the log file only gives you a few lines, not the full prompt. To extract the complete prompt, use Python:

```python
import dspy

# Load the saved optimized program
program = dspy.SokobanProgram()
program.load("outputs/gepa_<timestamp>/optimized_program.json")
print(program.solver.signature.instructions)
```

Or extract from the GEPA state directly:
```python
import pickle
with open("outputs/gepa_<timestamp>/gepa_logs/gepa_state.bin", "rb") as f:
    state = pickle.load(f)
# Inspect state for Pareto front programs and their instructions
```

The script also saves the best prompt to `outputs/gepa_<timestamp>/system_prompt.txt` automatically if the run completes.

## References

- [DSPy Documentation](https://dspy.ai/)
- [Verifiers Library](https://github.com/PrimeIntellect-ai/verifiers)
- [Reasoning Gym](https://github.com/open-thought/reasoning-gym)
- [vLLM](https://github.com/vllm-project/vllm)
