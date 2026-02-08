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

## References

- [DSPy Documentation](https://dspy.ai/)
- [Verifiers Library](https://github.com/PrimeIntellect-ai/verifiers)
- [Reasoning Gym](https://github.com/open-thought/reasoning-gym)
- [vLLM](https://github.com/vllm-project/vllm)
