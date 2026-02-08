# GEPA Progress Notes (Feb 7 2026)

## Critical Finding 1: PrimeRLAdapter Not Used by DSPy

**Root Cause**: Setting `lm.adapter = PrimeRLAdapter()` does NOT work.
DSPy's `Predict.forward()` uses `settings.adapter or ChatAdapter()` — it reads from
the global settings, NOT from the LM instance.

**Fix**: Use `dspy.configure(lm=student_lm, adapter=PrimeRLAdapter())`

**Evidence**: Debug script `debug_dspy_messages.py` showed:
- DSPy history had `[[ ## question ## ]]` field markers (default ChatAdapter)
- PrimeRLAdapter.format() was never called
- Model produced `[[ ## answer ## ]]` markers in response (ChatAdapter behavior)

## Critical Finding 2: Dataset Creation Path Matters

**Root Cause**: `verifiers.ReasoningGymEnv` creates a combined dataset of 5000+N items
and uses indices 5000..5000+N for eval. This produces different (harder) puzzles than
`rg.create_dataset("sokoban", size=N)` which uses indices 0..N.

**Impact**: GEPA val baseline was 12.5% with verifiers path vs 40% with direct rg.create_dataset.

**Fix**: Changed `load_datasets()` to use `rg.create_dataset()` directly instead of
going through `sokoban_env.load_environment()` → `vf.ReasoningGymEnv`.

**Evidence**: `diagnose_gap.py` showed DSPy and standalone eval produce identical results
(both 45%) on the same dataset. The discrepancy was entirely due to different puzzle
instances from different index ranges.

## Score Discrepancy Explained

| Method | Score | Explanation |
|--------|-------|-------------|
| Standalone eval (raw format, direct API) | 50.8% (127/250) | Model reasons at length, no scaffolding |
| GEPA Run 3 val (ChatAdapter, DSPy) | 12.6% (63/500) | Model produces short formatted response, no reasoning |
| GEPA Run 2 val (ChatAdapter, DSPy, temp=0.7) | 19.6% | Lower temp helps |
| GEPA Run 4 baseline test (PrimeRLAdapter, DSPy) | 36% (18/50) | Raw format with reasoning, matches standalone |
| GEPA Run 4 val (verifiers dataset, PrimeRLAdapter) | 12.5% (200 ex) | Harder puzzles from indices 5000+ |
| GEPA Run 5 baseline (direct rg dataset, PrimeRLAdapter) | 30% (50 ex) | Correct dataset + correct adapter |

The ChatAdapter format causes the model to produce very short responses (23 tokens)
with `[[ ## answer ## ]]` markers, skipping reasoning entirely. The raw format
allows the model to reason extensively (5000+ tokens) before answering.

## Adapter Fix Details

Made PrimeRLAdapter **signature-aware**:
- For `SokobanSolver` signature: raw system + user messages (matches prime-rl)
- For all other signatures (GEPA reflection etc.): falls back to ChatAdapter with field markers

This ensures GEPA's reflection pipeline (which uses its own DSPy signatures like
`GenerateEnhancedMultimodalInstructionFromFeedback`) still works correctly with
Claude Opus while the student model sees prime-rl-compatible format.

```python
# CORRECT: set adapter globally
adapter = PrimeRLAdapter()
dspy.configure(lm=student_lm, adapter=adapter)

# WRONG: this does NOT work - DSPy ignores lm.adapter
student_lm.adapter = PrimeRLAdapter()
dspy.configure(lm=student_lm)
```

## Critical Finding 3: DSPy Disk Cache Corrupts Cross-Run Evaluations

**Root Cause**: DSPy persistently caches LM responses on disk at `$HOME/.dspy_cache/`
(202MB SQLite database via `diskcache.FanoutCache`). Even at `temperature>0`, DSPy caches
the first stochastic result and returns the same sample for identical requests across runs.

**Cache key**: SHA256 of `{model, messages, temperature, max_tokens, _fn_identifier}`.
Ignores `api_key` and `api_base`. So stale entries from previous runs with different
API configs would still be returned.

**Impact**: Run 5's baseline showed 12% instead of true ~40% because the cache had
stale entries from previous diagnostic scripts that happened to produce bad results.

**Fix**: Clear DSPy cache in shell script BEFORE Python starts:
```bash
rm -rf "$HOME/.dspy_cache"
mkdir -p "$HOME/.dspy_cache"
```
**Cannot clear in Python**: DSPy opens the cache database on `import dspy`. Attempting
to clear it in Python after import causes "unable to open database file" errors because
DSPy still holds open file handles.

**Evidence**:
- `cache=False` on 20 val examples: 40% (8/20) — matches standalone
- Fresh `cache=True` on 20 val examples: 40% (8/20) — also matches
- Stale `cache=True` (Run 5 state): 12% — corrupted by old entries

## Critical Finding 4: GEPA with_instructions() Breaks Signature Identity

**Root Cause**: GEPA's `build_program()` calls `pred.signature.with_instructions(text)` which
creates a NEW signature class with `__name__ = "StringSignature"` instead of the original
`"SokobanSolver"`. The PrimeRLAdapter was checking `signature.__name__` to decide whether
to use raw format → GEPA's evaluations silently fell back to ChatAdapter format.

**Impact**: GEPA's base program val score was 13.5% (ChatAdapter format) instead of ~37%
(raw format). GEPA was essentially optimizing prompts while evaluating them in the wrong
format. The baseline test (34%) used the original program directly (without `build_program()`),
so it worked correctly — masking the bug.

**Fix**: Changed `_is_raw_format_signature()` to check field names instead of `__name__`:
```python
# OLD (broken): checks __name__ which changes after with_instructions()
name = getattr(signature, '__name__', '') or ''
return name in self.RAW_FORMAT_SIGNATURES  # {"SokobanSolver"}

# NEW (fixed): checks field structure which is preserved
input_keys = frozenset(signature.input_fields.keys())
output_keys = frozenset(signature.output_fields.keys())
return (input_keys, output_keys) == (frozenset({"question"}), frozenset({"answer"}))
```

**Evidence**:
- Run 6 base program val: 13.5% (27/200) — StringSignature → ChatAdapter
- Run 7 base program val: 37% (74/200) — StringSignature → raw format (fixed!)
- Baseline test_set[:50]: 28-34% — original program → raw format (always worked)

## Run 7 Final Results (First Correctly-Working GEPA Run)

- **Status**: Completed (allocation expired during iteration 15's subsample eval)
- **Output dir**: `outputs/gepa_20260207_190039/`
- **Log**: `/pscratch/sd/s/siddart2/dspy_gepa/gepa_opus_run7.log`
- **Run script**: `run_gepa_opus_v7.sh` (with cache clearing + field-based adapter fix)
- **Settings**: train=500, val=200, test=500, temp=1.0, max_tokens=8192, minibatch=5
- **Dataset**: direct rg.create_dataset (same as Run 5/6)
- **Cache**: Cleared before run (fresh)
- **Duration**: 15 iterations, 940/10000 rollouts (9%), ~47 minutes

### Pareto Front Programs (4 programs)

| Program | Iteration | Val Score | Prompt |
|---------|-----------|-----------|--------|
| 0 | baseline | **37%** (74/200) | Original SokobanSolver docstring (generic reasoning + `<answer>` format) |
| 1 | 6 | 16% (32/200) | "You are going to solve a 'sokoban' puzzle. Your goal is to push all boxes onto all goal positions." |
| 2 | 10 | **38%** (76/200) | **"You are going to solve a 'sokoban' puzzle."** |
| 3 | 14 | 36.5% (73/200) | Long detailed Sokoban strategy prompt (symbol legend, rules, strategy, ~80 lines) |

**Best single program**: Program 2 at 38% — simple one-liner
**Pareto front aggregate**: **55.5%** (111/200) — best across all 4 programs per example

### Best GEPA-Optimized Prompt (Program 2)

```
You are going to solve a 'sokoban' puzzle.
```

This replaces the original multi-line docstring with a simple task identifier.
The 1% improvement (37% → 38%) is within noise at temperature=1.0.

### Key Insight: Prompt Length Hurts Performance

Program 3 (the detailed strategy prompt, ~80 lines) scored **lower** (36.5%) than the
simple one-liner (38%) despite containing explicit Sokoban rules, strategy tips, and
common mistakes. The reason: the long system prompt consumes tokens from the 8192-token
budget, causing more responses to be truncated before producing `<answer>` tags.
The model already knows how to solve Sokoban from its training — it doesn't need
instructions about the rules.

### High Stochastic Variance at Temperature=1.0

The Pareto aggregate of 55.5% vs best single-program 38% shows that different prompts
solve different subsets of examples. With temperature=1.0 and single-sample evaluation,
there's significant variance. The true capability of the model is probably in the 35-40%
range, but which specific examples it solves varies between runs.

### Baseline Test
- Baseline test_set[:50]: 28% (14/50) — within expected range (high variance on small N)

## Invalidated Runs

- **Run 5**: Stopped due to cache corruption (see Critical Finding 3)
  - Output dir: `outputs/gepa_20260207_155638/` — results are INVALID
- **Run 6**: Stopped due to adapter bug (see Critical Finding 4)
  - Output dir: `outputs/gepa_20260207_183200/` — results are INVALID
  - GEPA evaluated programs in ChatAdapter format despite PrimeRLAdapter being set
  - The baseline test was correct (used original program), but GEPA's internal evals were wrong

## Key Observations

1. **Thinking mode**: Qwen3-4B generates reasoning in plain text (no `<think>` tags)
   even though it's the Instruct variant. The vLLM server wasn't started with
   `--enable-thinking` flag but the model still reasons extensively.

2. **Truncation**: Many responses get truncated at max_tokens=8192 before producing
   `<answer>` tags. rl.toml also uses max_tokens=8192, so this matches training.

3. **Per-example timing**: ~2 sec per example with raw format (vs ~0.01 sec with ChatAdapter).
   Full val eval of 200 examples takes ~6 min.

4. **GEPA reflection issue**: When GEPA proposes new instructions for SokobanSolver,
   it replaces the entire docstring (which includes `<answer>` format instructions).
   The new prompts often omit the `<answer>` tag format, causing extraction issues.
   The metric's feedback mechanism should help GEPA learn to include format instructions.

## Run Results History

| Run | Reflection | MiniBatch | Temp | Adapter | Dataset | Val | Test | Notes |
|-----|-----------|-----------|------|---------|---------|-----|------|-------|
| 1 | Claude Opus | 3 | 0.7 | ChatAdapter (default) | verifiers | 13.6% | 13.6% | First run |
| 2 | Claude Opus | 5 | 0.7 | ChatAdapter (default) | verifiers | 19.6% | 20.8% | Best so far |
| 3 | Claude Opus | 5 | 1.0 | ChatAdapter (bug!) | verifiers | 12.6% stuck | - | Adapter not used |
| 4 | Claude Opus | 5 | 1.0 | PrimeRLAdapter (fixed) | verifiers | 12.5% base | - | Hard puzzles from idx 5000+ |
| 5 | Claude Opus | 5 | 1.0 | PrimeRLAdapter (fixed) | direct rg | 12% INVALID | - | Cache corruption |
| 6 | Claude Opus | 5 | 1.0 | PrimeRLAdapter (field-name bug!) | direct rg | 13.5% val | 34% base | with_instructions broke adapter |
| 7 | Claude Opus | 5 | 1.0 | PrimeRLAdapter (fully fixed) | direct rg | **38% best** | 28% base | First correct run, 55.5% Pareto agg |

## Files

- `sokoban_gepa.py` — Main GEPA script with PrimeRLAdapter (signature-aware), direct rg.create_dataset
- `eval_prompt_standalone.py` — Standalone eval (bypasses DSPy, direct API calls)
- `run_gepa_opus_v7.sh` — Template run script (cache clearing + all bug fixes)
- `start_vllm_server.sh` — vLLM server startup script
- `PROGRESS_NOTES.md` — This file
- `README.md` — Project documentation
- `gepa_opus_run7.log` — Run 7 log (first correct run)
- `outputs/gepa_20260207_190039/` — Run 7 output directory

## Overall Conclusions

### GEPA's Value for This Task

After 7 runs and fixing 4 critical bugs, GEPA produced a best single-program improvement
of only **1% absolute** (37% → 38%) over the baseline prompt. The Pareto aggregate of
55.5% is misleading — it represents an unrealizable oracle that picks the best program
per-example, which isn't useful for a single context distillation prompt.

**For Sokoban with Qwen3-4B at temperature=1.0, the system prompt has minimal impact
on performance.** The model's reasoning ability (and whether it can finish reasoning within
8192 tokens) is the primary bottleneck.

### Recommended System Prompt for prime-rl Context Distillation

Use either:
1. **Simple**: `"You are going to solve a 'sokoban' puzzle."` (GEPA's best, 38%)
2. **Original**: The default verifiers system prompt (baseline, 37%)

Both perform equivalently within noise. The simple prompt is shorter and leaves more
tokens for reasoning.

### What Would Help More Than Prompt Optimization

1. **Increase max_tokens** beyond 8192 — many responses are truncated mid-reasoning
2. **Lower temperature** (0.7 vs 1.0) — reduces variance, Run 2 showed 19.6% vs 12.6%
   at temp=0.7 vs 1.0 (with ChatAdapter, so not directly comparable, but the trend holds)
3. **Larger GEPA minibatch** — minibatch=5 causes high-variance subsample evaluation,
   most proposed prompts get rejected based on noisy 5-example samples
4. **More compute time** — Run 7 only completed 9% of the 10K rollout budget; a longer
   run might discover more diverse prompts

### How to Resume GEPA Run 7

GEPA saves state to `outputs/gepa_20260207_190039/gepa_logs/gepa_state.bin`.
However, DSPy GEPA doesn't natively support resume-from-checkpoint.
To continue, you would need to:
1. Get a new allocation and start vLLM server
2. Re-run with the same settings (the GEPA state.bin may be loadable)
3. Or start a fresh Run 8 with the same fixed code

## Important Lessons for Future

1. **DSPy adapter must be set via `dspy.configure(adapter=...)`**, NOT `lm.adapter`
2. **Custom adapters must be signature-aware** if using GEPA (which has its own signatures)
3. **ChatAdapter makes Qwen3 skip reasoning** — the structured format causes very short outputs
4. **Match ALL sampling settings** from rl.toml (temperature, max_tokens) for valid comparison
5. **rl.toml seq_len (16884) ≠ max_tokens (8192)**: seq_len is total sequence, max_tokens is completion only
6. **Dataset creation path matters**: `verifiers.ReasoningGymEnv` uses index offset of `num_train_examples`
   for eval data, producing different puzzles than direct `rg.create_dataset`. Always use direct creation
   for consistent comparison with standalone eval.
7. **GEPA subsample size (minibatch) is critical**: With minibatch=5, many iterations are skipped
   because the subsample doesn't show improvement. Larger minibatch = more reliable but slower.
8. **DSPy disk cache must be cleared between runs**: `rm -rf $HOME/.dspy_cache && mkdir -p $HOME/.dspy_cache`
   Must be done in shell script BEFORE Python starts (not in Python code after import).
9. **DSPy caches stochastic results at temperature>0**: Returns the same stochastic sample
   for identical requests across runs, corrupting evaluations.
10. **GEPA absolute scores at temperature=1.0 may be unreliable**: With caching, each
    (prompt, example) pair is evaluated once. At temperature=1.0, single-sample evaluations
    have high variance. GEPA's relative comparisons between programs may still be meaningful,
    but absolute scores can vary wildly from standalone eval.
11. **DSPy `with_instructions()` changes `__name__`**: Calling `signature.with_instructions(text)`
    creates a new `StringSignature` class, losing the original class name. Never check
    `__name__` for signature identity — use field names instead.
12. **Baseline tests can mask GEPA evaluation bugs**: The baseline evaluation uses the
    original program directly, while GEPA's evaluations go through `build_program()`.
    If `build_program()` changes something (like signature identity), the baseline test
    may pass while GEPA's evaluations silently break.
13. **Longer prompts can hurt performance**: With fixed max_tokens, a longer system prompt
    leaves fewer tokens for reasoning. GEPA's most detailed prompt (80 lines of strategy)
    scored lower (36.5%) than a simple one-liner (38%). The model already knows the task.
14. **GEPA Pareto aggregate is misleading for single-prompt tasks**: The Pareto front
    aggregate (55.5%) represents an oracle that picks the best program per-example.
    For context distillation where you need one prompt, only the best single program
    score matters (38%). The aggregate inflates perceived improvement.
15. **Temperature=1.0 makes GEPA optimization noisy**: With single-sample evaluation
    at high temperature, the same prompt can score 28-38% across different runs.
    GEPA's minibatch filter (5 examples) rejects many good prompts due to noise.
    Consider temperature=0.7 or multiple samples per evaluation for more stable GEPA.
