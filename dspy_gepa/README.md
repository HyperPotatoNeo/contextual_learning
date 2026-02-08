# GEPA Prompt Optimization for Sokoban

This project uses [GEPA (Genetic-Pareto)](https://dspy.ai/) prompt optimization from DSPy to optimize system prompts for solving Sokoban puzzles with Qwen3-4B, targeting [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) context distillation.

## Quick Start

```bash
# 1. Start vLLM server (in container on compute node)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --tensor-parallel-size 4 \
    --max-model-len 16384 \
    --port 8000

# 2. Run GEPA (use run_gepa_opus_v7.sh as template!)
./run_gepa_opus_v7.sh
```

**CRITICAL**: Always use `run_gepa_opus_v7.sh` as the template for new runs. It includes cache clearing and all bug fixes. See "Known DSPy Pitfalls" below.

## Results (Run 7 — First Correct Run)

| Program | Val Score (200 ex) | Prompt |
|---------|-------------------|--------|
| 0 (baseline) | 37% | Generic reasoning prompt with `<answer>` format |
| 2 (best) | **38%** | `"You are going to solve a 'sokoban' puzzle."` |
| 3 | 36.5% | Detailed 80-line strategy prompt |

**Pareto aggregate**: 55.5% (oracle across all programs per example)

**Conclusion**: For Sokoban with Qwen3-4B at temperature=1.0, the system prompt has minimal impact on performance. The model's reasoning ability within the 8192-token budget is the bottleneck, not the prompt.

## Architecture

```
dspy_gepa/
├── sokoban_gepa.py          # Main GEPA script (PrimeRLAdapter, direct rg.create_dataset)
├── eval_prompt_standalone.py # Standalone eval (bypasses DSPy, direct API calls)
├── run_gepa_opus_v7.sh      # Run script template (cache clearing + all fixes)
├── PROGRESS_NOTES.md        # Detailed progress notes with all findings
├── debug_dspy_messages.py   # Debug: what messages DSPy sends
├── debug_dspy_eval.py       # Debug: DSPy eval vs standalone comparison
├── diagnose_gap.py          # Diagnostic: DSPy vs standalone on same examples
├── test_adapter_fix.py      # Test: adapter fix verification
├── test_dataset_fix.py      # Test: dataset creation fix verification
├── test_threading_impact.py # Test: 1-thread vs 32-thread eval
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

python -u sokoban_gepa.py \
    --reflection-model claude-opus \
    --train-size 500 --val-size 200 --test-size 500 \
    --max-metric-calls 10000 \
    --reflection-minibatch-size 5 \
    --num-threads 32 \
    --temperature 1.0 \
    --max-tokens 8192
```

### Standalone Evaluation (No DSPy)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

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

## Known DSPy Pitfalls

Four critical bugs were discovered and fixed during development. **All fixes are in `sokoban_gepa.py` and `run_gepa_opus_v7.sh`**, but understanding them prevents future issues:

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

## References

- [DSPy Documentation](https://dspy.ai/)
- [Verifiers Library](https://github.com/PrimeIntellect-ai/verifiers)
- [Reasoning Gym](https://github.com/open-thought/reasoning-gym)
- [vLLM](https://github.com/vllm-project/vllm)
