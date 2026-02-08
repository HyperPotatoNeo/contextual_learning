# GEPA Looped Context Distillation: Experiment Plan

## Table of Contents

1. [What to Expect from Context Distillation](#1-what-to-expect-from-context-distillation)
2. [Frontier Lab Hyperparameter Reference](#2-frontier-lab-hyperparameter-reference)
3. [On-Policy Distillation Literature](#3-on-policy-distillation-literature)
4. [Code Review: Is Distillation Done Right?](#4-code-review-is-distillation-done-right)
5. [GEPA Findings and Implications](#5-gepa-findings-and-implications)
6. [Experiment Plan](#6-experiment-plan)
7. [Recommended Configs](#7-recommended-configs)
8. [References](#8-references)

---

## 1. What to Expect from Context Distillation

### 1.1 What Context Distillation Is

Context distillation uses the **same model** with **different prompts**:
- **Teacher**: model + enhanced system prompt (detailed instructions, strategies, examples)
- **Student**: model + minimal/no system prompt

The student generates rollouts on-policy, the teacher scores them via logprob computation with the enhanced context prepended, and the student learns to internalize the teacher's context-induced behavior through reverse-KL minimization.

This is distinct from standard knowledge distillation (larger teacher -> smaller student). In context distillation, the model distills its own contextual knowledge into its weights.

### 1.2 Expected Performance Gains

Based on the literature and our GEPA results:

| Setting | Expected Improvement | Source |
|---------|---------------------|--------|
| Same model, enhanced prompt -> base prompt | 0-5% absolute | GEPA Run 7, Anthropic |
| Larger teacher -> smaller student (on-policy) | 5-15% absolute | GKD, Thinking Machines |
| RL-trained teacher -> base student (on-policy) | 10-20% absolute | SkyRL, Qwen3 |
| RL-trained teacher -> base student (off-policy SFT) | 5-10% absolute | DeepSeek-R1 Stage 3 |

**For our Sokoban setting** (Qwen3-4B, same model context distillation):
- GEPA's best prompt (structured ~40 lines) scored 38% vs 37% baseline (within noise at temp=1.0)
- The model already knows Sokoban rules from pretraining -- the bottleneck is reasoning within the 8192-token budget, not instruction quality
- Context distillation will likely yield **1-5% improvement** at best for pure prompt-based teacher
- **Larger gains require a stronger teacher**: either a larger model or an RL-trained checkpoint

### 1.3 When Context Distillation Helps Most

Context distillation works best when:
1. The teacher context provides **genuinely useful information** the model can't infer from the base prompt
2. The model has **sufficient capacity** to learn the behavior (not limited by model size)
3. The **token budget** is large enough for the model to reason (truncation kills performance)
4. The task has **verifiable outcomes** so you can measure actual improvement

For Sokoban, the main value proposition is: can the student learn to replicate the teacher's reasoning patterns (e.g., grid parsing, deadlock detection, systematic push planning) without needing the verbose system prompt?

### 1.4 The Looped Distillation Concept

**GEPA Looped Distillation** extends context distillation with an iterative prompt optimization loop:

1. **GEPA optimizes the teacher prompt** using evolutionary search over prompt variants
2. **Context distillation** trains the student to internalize the best teacher prompt
3. **Evaluate** the distilled student
4. **(Optional) Repeat**: use the distilled student as a new baseline and optimize a new teacher prompt

The hypothesis: GEPA finds prompts that are not just marginally better for the teacher, but that produce reasoning patterns the student can **most easily internalize** via distillation. The optimal teacher prompt for distillation may differ from the optimal prompt for direct performance.

---

## 2. Frontier Lab Hyperparameter Reference

### 2.1 DeepSeek-R1 (January 2025)

**Source**: arXiv:2501.12948 -- "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"

Uses a multi-stage pipeline: Cold Start SFT -> Reasoning RL (GRPO) -> Rejection Sampling + SFT -> General RL.

| Hyperparameter | Value | Notes |
|---|---|---|
| Algorithm | GRPO | Group Relative Policy Optimization |
| Learning rate | ~1e-6 | |
| KL coefficient | ~0.001, annealed | Gradually reduced over training |
| Clip range | ~10 (not 0.2) | Much wider than standard PPO |
| Temperature | 1.0 | For rollout generation |
| Group size (rollouts/prompt) | 64 | Very large groups |
| Max response length | 8K -> 32K | Progressive extension |
| Batch size | 512 prompts/step | ~32K sequences total |
| Reward | Rule-based (math, code) | No learned reward model |

**GRPO specifics** (from DeepSeek-Math, arXiv:2402.03300):

| Hyperparameter | Value |
|---|---|
| Learning rate | 1e-6 |
| KL coefficient (beta) | 0.04 |
| Group size | 64 |
| Temperature | 0.6 (eval: 0.0) |
| Batch size | 512 prompts |
| Max tokens | 2048 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |

### 2.2 DAPO (March 2025)

**Source**: arXiv:2503.14476 -- "DAPO: An Open-Source LLM Reinforcement Learning System" (ByteDance)

Key innovations: asymmetric clipping, no KL penalty, dynamic sampling, token-level loss, overlong reward shaping.

| Hyperparameter | Value | Notes |
|---|---|---|
| Base model | Qwen2.5-32B | |
| Learning rate | 1e-6 | Constant schedule |
| KL penalty | **0.0** | Removed entirely |
| Clip epsilon (positive adv) | 0.28 | Wider for positive updates |
| Clip epsilon (negative adv) | 0.2 | Tighter for negative updates |
| Temperature | 1.0 | |
| Top-p | 0.7 | |
| Group size | 16 | |
| Batch size | 512 prompts | |
| Mini-batch size | 512 | |
| Max tokens | 20,480 | |
| Training steps | ~18K | |
| Entropy bonus | 0.001 | Token-level |
| Overlong reward | Soft -1.0 beyond max length | |

**DAPO ablation results** (each technique's contribution):

| Technique | AIME 2024 Score |
|---|---|
| GRPO baseline | 30/30 (from their report) |
| + Clip-Higher (asymmetric clip) | +4-6 points |
| + Dynamic Sampling | +2-3 points |
| + Token-Level Loss | +1-2 points |
| + Overlong Reward Shaping | +1-2 points |
| Full DAPO | ~50/30 equivalent |

### 2.3 JustRL (2025)

**Source**: Blog post -- "Just Use RL: A Simple Recipe That Rivals DeepSeek-R1"

Demonstrates that a simple fixed-hyperparameter GRPO recipe matches complex approaches with 2x less compute.

| Hyperparameter | Value | Notes |
|---|---|---|
| Learning rate | 1e-6 | Constant (no warmup, no decay) |
| KL penalty | 0.0 | No KL loss |
| Entropy regularization | None | |
| Clip range | [0.8, 1.28] | Asymmetric like DAPO |
| Temperature | 1.0 | |
| Batch size | 256 prompts | |
| Rollouts per prompt | 8 | |
| Max tokens | 15,000 | |
| Training steps | ~4,000 | |
| Model | Qwen2.5-32B | |

### 2.4 Qwen3 (May 2025)

**Source**: arXiv:2505.09388 -- "Qwen3 Technical Report"

Four-stage pipeline: Long-CoT Cold Start -> Reasoning RL -> Thinking Mode Fusion -> General RL.

| Hyperparameter | Value | Notes |
|---|---|---|
| Algorithm | GRPO | |
| Reasoning RL stages | Progressive length 8K -> 32K | |
| Tasks | Math, code, logic, science | Rule-based rewards |
| General RL | GRPO + reward models | Hybrid rule-based and learned |
| On-policy distillation | Used in pipeline | Cited as more compute-efficient than RL |

Qwen3 explicitly states that on-policy distillation is a "more compute efficient approach than RL" for certain post-training tasks.

### 2.5 Llama 3.1 (July 2024)

**Source**: arXiv:2407.21783 -- "The Llama 3 Herd of Models"

Classic PPO-RLHF (not GRPO):

| Hyperparameter | Value |
|---|---|
| Algorithm | PPO |
| Learning rate | ~1e-6 |
| Clip epsilon | 0.2 |
| Batch size | 512 prompts |
| Max tokens | 4096 |
| GAE lambda | 1.0 (REINFORCE) |
| DPO beta (later stage) | 0.1 |
| DPO learning rate | 1e-7 |
| Iterative DPO rounds | 6 |

### 2.6 Cross-Method Summary

| Hyperparameter | DeepSeek-R1 | DAPO | JustRL | Qwen3 | Our Config |
|---|---|---|---|---|---|
| **LR** | 1e-6 | 1e-6 | 1e-6 | ~1e-6 | **1e-5** |
| **KL coeff** | 0.001->0 | 0.0 | 0.0 | Small | **0.0** (via tau) |
| **Clip/Mask** | ~10 | 0.2/0.28 | 0.8/1.28 | -- | **0.125/8.0** |
| **Temperature** | 1.0 | 1.0 | 1.0 | 1.0 | **1.0** |
| **Group size** | 64 | 16 | 8 | 8-16 | **16** |
| **Batch** | 512 | 512 | 256 | -- | **256** |
| **Max tokens** | 32K | 20K | 15K | 32K | **8192** |
| **Weight decay** | 0.1 | -- | -- | -- | **0.0** |
| **Gradient clip** | 1.0 | 1.0 | -- | -- | **1.0** |

**Key observation**: Our learning rate of 1e-5 is **10x higher** than the RL consensus of 1e-6. This is intentional for distillation (stronger, denser gradient signal), but worth ablating.

---

## 3. On-Policy Distillation Literature

### 3.1 GKD: Generalized Knowledge Distillation (Google DeepMind, 2023)

**Source**: arXiv:2306.13649 -- "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"

**The most important paper for understanding on-policy distillation design choices.**

Key findings from ablations:
1. **On-policy vs off-policy is the single most important factor** -- even forward KL with on-policy data beats reverse KL with off-policy data
2. **Divergence type matters less than data distribution**: Forward KL, reverse KL, and JSD all improve with on-policy data
3. **JSD (beta=0.5) often beats both pure forward and reverse KL** -- balances mode-covering and mode-seeking
4. **50% on-policy mixing often matches 100% on-policy** -- some off-policy diversity helps

Results (PaLM 2 distillation, GSM8K):
- Standard KD (off-policy, forward KL): ~40%
- On-policy GKD (reverse KL): ~48%
- **Improvement: +8% absolute**

### 3.2 MiniLLM (2023)

**Source**: arXiv:2306.08543 -- "Knowledge Distillation of Large Language Models"

Frames distillation as RL: reward = `log p_teacher(t) - log p_student(t)` per token.

| Parameter | Value |
|-----------|-------|
| LR | 1e-5 to 5e-6 |
| Batch size | 32 |
| Max seq len | 256-512 |
| Training steps | 20K-40K |
| Reward baseline | Running mean |
| Clipping | None (plain REINFORCE) |

Key finding: Reverse KL + on-policy gives **~12% relative improvement** over forward KL + teacher-forced.

### 3.3 Thinking Machines / Tinker On-Policy Distillation

**Source**: thinkingmachines.ai blog

Results on AIME'24:
- On-policy distillation: **74.4%** with 1,800 GPU hours
- Standard RL training: 67.6% with 17,920 GPU hours
- **10x cheaper for better results**

Architecture: student generates, teacher scores, reverse KL as reward, importance sampling loss (no clipping).

### 3.4 SkyRL/NovaSky On-Policy Distillation

**Source**: SkyRL codebase at `/pscratch/sd/s/siddart2/SkyRL/`

Exact hyperparameters from `run_on_policy_distill_math_qwen3_4b.sh`:

| Parameter | Value |
|-----------|-------|
| Student | Qwen/Qwen3-4B-Base |
| Teacher | RL-trained Qwen3-4B (DAPO step 90) |
| LR | 1e-5 |
| Weight decay | 0.1 |
| Temperature (train) | 1.0 |
| Top-p (train) | 1.0 |
| Top-p (eval) | 0.7 |
| Batch size | 512 |
| Rollouts per prompt | 16 |
| Max generate | 8192 |
| Epochs | 20 |
| Advantage estimator | `no_op` (passthrough) |
| Policy loss | `importance_sampling` (no clipping) |
| KL in reward | True |
| KL loss | False |
| Max prompt length | 2048 |
| Warmup steps | 0 |

The KL reward computation:
```
rewards = -(student_logprob - teacher_logprob) * loss_mask
```

This is the per-token reverse KL, used directly as the reward signal with no baseline subtraction (advantage_estimator = no_op). The importance sampling loss:
```
loss = -exp(log_probs - old_log_probs) * advantages
```

### 3.5 OPSD: Self-Distilled Reasoner (2025)

**Source**: arXiv:2505.XXXXX -- "OPSD: On-Policy Self-Distillation"

Single model serves as both teacher and student with different conditioning (e.g., privileged information).

| Parameter | Value |
|-----------|-------|
| LR | 2e-5 |
| Batch size | 32 effective |
| LoRA rank | 64 |
| Max completion | 2048 tokens |
| Temperature | 1.2 |
| Divergence | JSD (beta=0.5) |

**4-8x more token-efficient than GRPO.** Uses JSD instead of pure reverse KL.

### 3.6 Sebastian Raschka's State of LLM Reasoning Training (2025)

Key insights from this overview:
- On-policy distillation is emerging as the **preferred method** for transferring reasoning abilities
- The field is converging on: GRPO-style advantage, reverse KL, on-policy sampling, no clipping
- Temperature 1.0 for training, lower for evaluation is nearly universal
- The trend is toward **removing KL penalties** and relying on importance ratio masking instead

### 3.7 Forward KL vs Reverse KL

| Property | Forward KL | Reverse KL |
|----------|-----------|------------|
| Optimizes | `E_teacher[log(p_student/p_teacher)]` | `E_student[log(p_student/p_teacher)]` |
| Behavior | Mode-covering (mass-spreading) | Mode-seeking (peak-finding) |
| Samples from | Teacher distribution | Student distribution (on-policy) |
| Failure mode | Student spreads probability, bland output | Student collapses to few modes |
| Top-k quality | Worse (thin spread) | Better (concentrated) |

**Consensus**: Reverse KL is preferred for task-specific distillation. JSD is preferred when diversity matters.

---

## 4. Code Review: Is Distillation Done Right?

### 4.1 Summary

After reading all relevant source files in prime-rl, the implementation is **correct** for the current setting. The core distillation pipeline:

1. Student generates rollouts (on-policy via vLLM) -- **correct**
2. Teacher scores completions via prefill with enhanced context -- **correct**
3. Reverse KL computed as reward signal -- **correct**
4. Advantage centered within problem groups via `use_full_reward_baseline` -- **correct**
5. Policy gradient update with importance ratio masking -- **correct**

### 4.2 What Exactly Is Being Optimized

With the config values `adv_tau=0.0, teacher_tau=1.0, student_tau=1.0, use_full_reward_baseline=true`:

```
full_reward_i = 0 * task_reward + 1.0 * sum(teacher_lp) - 1.0 * sum(inference_lp)
              = sum_t [log p_teacher(t) - log p_student(t)]
              = -KL(student || teacher)  (sequence-level)

advantage_i = full_reward_i - mean(full_reward) per problem group
```

Then in the loss:
```
coeff = importance_ratio * advantage    (kl_tau=0 so no extra KL term)
loss = -sum_t [coeff.detach() * trainer_logprobs_t]  over keep_mask tokens
```

This is a REINFORCE-style policy gradient where:
- Sequences where student matches teacher well -> positive advantage -> student encouraged to generate similar sequences
- Sequences where student diverges -> negative advantage -> student discouraged

**This is correct reverse-KL on-policy context distillation.**

### 4.3 Key Implementation Details

**Teacher logprob computation** (`orchestrator/utils.py:175-245`):
- Correctly constructs `[system: teacher_context, user: prompt_text]`
- Applies chat template to get `teacher_prompt_ids`
- Concatenates `teacher_prompt_ids + completion_ids` for vLLM prefill
- Slices `all_logprobs[len(teacher_prompt_ids):]` to get completion-only logprobs
- **No off-by-one errors** -- alignment is correct

**Logprob padding** (`orchestrator/orchestrator.py:427-429`):
- Prompt positions get 0.0 logprobs (both teacher and inference)
- These positions are masked out by `loss_mask` in the loss -- **correct**

**Config sync** (`rl.py:371-384`):
- `auto_setup_advantage_tau` validator copies tau values from `trainer.loss` to `orchestrator.advantage`
- Sets `use_full_reward_baseline` flag on both sides -- **correct**

### 4.4 Issues Found

| # | Severity | Description | Impact |
|---|----------|-------------|--------|
| 1 | LOW | `_extract_prompt_text` drops existing system messages from chat-format prompts | Not a problem for Sokoban (simple user messages) |
| 2 | MEDIUM | With branching rollouts + `use_full_reward_baseline`, only first example's logprobs are used per rollout | Current config uses interleaved (default), so no impact |
| 3 | LOW | `skip_verification` not set despite `adv_tau=0.0` -- wastes compute on unused task rewards | Intentional for monitoring reward in W&B |
| 4 | MEDIUM | `use_full_reward_baseline` defaults to `True` in both configs -- could trap standalone users who set `adv_tau!=1.0` without using `rl.py` | Not relevant when running through `rl.py` |

### 4.5 Scalar vs Per-Token Advantage

When `use_full_reward_baseline=True`, the advantage is a **scalar per sequence** (same value for every token). The non-full-reward-baseline path adds per-token KL terms (`teacher_tau * teacher_lp_t - student_tau * trainer_lp_t`), providing token-level differentiation.

This is a **design decision**, not a bug. The scalar advantage provides group-level signal about which sequences are better. Per-token differentiation comes from the importance ratio masking. The SkyRL implementation also uses scalar (sequence-level) advantage with the `no_op` advantage estimator.

---

## 5. GEPA Findings and Implications

### 5.1 GEPA Run 7 Results (First Correct Run)

| Program | Val Score | Prompt |
|---------|-----------|--------|
| Baseline | 37% (74/200) | Original SokobanSolver docstring |
| Best GEPA (Prog 2) | 38% (76/200) | Structured ~40-line prompt: symbols, rules, movement clarification, strategy, answer format |
| Detailed strategy (Prog 3) | 36.5% (73/200) | Longer ~80-line detailed solving guide |
| Pareto aggregate | 55.5% (111/200) | Best of 4 programs per example |

The best GEPA prompt (Program 2) includes concise symbol definitions, push mechanics,
a 7-point solving strategy, and critical output instructions (`<answer>` format).
It is notably shorter and more focused than Program 3 or the 202-line rl.toml teacher context.
See the full prompt in `dspy_gepa/PROGRESS_NOTES.md`.

**Key findings**:
1. **Prompt structure matters more than length** -- the medium-length structured prompt (38%) outperformed both the baseline and the verbose guide (36.5%)
2. **Overly long prompts hurt** -- they consume tokens from the 8192-token budget, causing more truncations
3. **The Pareto aggregate (55.5%) is misleading** -- it's an unrealizable oracle, not a single-prompt score
4. **High stochastic variance at temp=1.0** makes reliable prompt comparison difficult

### 5.2 Implications for Context Distillation

1. **The 202-line teacher context in rl.toml may be suboptimal**: GEPA's best prompt achieves the same score with ~40 lines. The teacher only does prefill (not generation) so token budget is less of a concern on the teacher side, but a more focused prompt may still produce better logprob signals.

2. **GEPA's best prompt could replace the rl.toml teacher context**: The structured ~40-line prompt covers symbols, rules, strategy, and answer format -- all the essential information without the verbosity. Consider using it as the teacher context for distillation.

3. **The real value of context distillation may not be in the prompt content**: the teacher's enhanced context allows it to assign more confident logprobs to correct reasoning steps. Even if the prompt doesn't change accuracy much, it may change the **distribution of logprobs** in ways that provide useful learning signal.

### 5.3 Critical Bugs Found in GEPA Integration

These were all fixed before Run 7 but are important lessons:

1. **DSPy adapter must be set via `dspy.configure(adapter=...)`**, not `lm.adapter`
2. **`with_instructions()` changes signature `__name__`** -- check field names instead
3. **DSPy disk cache corrupts cross-run evaluations** -- must clear before each run
4. **Dataset creation path matters** -- `verifiers.ReasoningGymEnv` uses different index ranges than `rg.create_dataset`

---

## 6. Experiment Plan

### 6.1 Phase 1: Baseline Context Distillation (Current Config)

**Goal**: Establish baseline performance of context distillation with the current config.

**Config**: Use `experiments/context_distill/rl.toml` as-is.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | Qwen3-4B-Instruct-2507 | Current model |
| LR | 1e-5 | Distillation consensus (higher than RL) |
| adv_tau | 0.0 | Pure distillation |
| teacher_tau | 1.0 | Full teacher signal |
| student_tau | 1.0 | Full entropy regularization |
| batch_size | 256 | |
| rollouts_per_example | 16 | Matches SkyRL |
| max_tokens | 8192 | Matches training |
| temperature | 1.0 | Standard |
| max_steps | 400 | |
| LoRA rank/alpha | 16/64 | |

**Expected outcome**: 0-5% improvement over baseline on Sokoban eval.

**Metrics to watch**:
- `teacher_kl` -- should decrease over training
- `reward/mean` -- should remain stable or increase (even with adv_tau=0)
- `is_masked` fraction -- should stay below 50%
- `geo_seq_ratio` -- should stay near 1.0 (on-policy)

### 6.2 Phase 2: Hyperparameter Ablations

Run each ablation for 400 steps and compare to Phase 1 baseline.

#### 6.2.1 Learning Rate

| Experiment | LR | Rationale |
|---|---|---|
| lr_1e6 | 1e-6 | RL consensus value, more conservative |
| lr_1e5 (baseline) | 1e-5 | Distillation consensus |
| lr_5e5 | 5e-5 | More aggressive, test stability |

**Hypothesis**: 1e-5 is correct for distillation. 1e-6 may be too slow for 400 steps. 5e-5 may cause instability.

#### 6.2.2 Teacher/Student Tau Ratio

| Experiment | teacher_tau | student_tau | Rationale |
|---|---|---|---|
| tau_1_1 (baseline) | 1.0 | 1.0 | Equal weight |
| tau_1_0 | 1.0 | 0.0 | No entropy bonus, pure teacher matching |
| tau_05_05 | 0.5 | 0.5 | Reduced distillation pressure |
| tau_01_01 | 0.01 | 0.01 | Very light distillation (almost no signal) |

**Hypothesis**: The student_tau term acts as entropy regularization. Setting it to 0 may cause mode collapse. Equal teacher/student tau is standard in the literature.

#### 6.2.3 Hybrid: Task Reward + Distillation

| Experiment | adv_tau | teacher_tau | student_tau | Rationale |
|---|---|---|---|---|
| pure_distill (baseline) | 0.0 | 1.0 | 1.0 | Pure distillation |
| hybrid_balanced | 0.5 | 0.5 | 0.5 | Balance task reward and distillation |
| hybrid_task_dominant | 1.0 | 0.1 | 0.1 | Mostly RL, light KL regularization |
| hybrid_distill_dominant | 0.1 | 0.8 | 0.8 | Mostly distillation, light task reward |

**Hypothesis**: For Sokoban, the task reward (binary correct/incorrect) combined with distillation should outperform either signal alone. The hybrid approach gives the model two learning signals: "match the teacher's reasoning" and "actually solve the puzzle."

#### 6.2.4 Temperature

| Experiment | Temperature | Rationale |
|---|---|---|
| temp_06 | 0.6 | DeepSeek-Math style, less exploration |
| temp_10 (baseline) | 1.0 | Standard for distillation |

**Hypothesis**: Lower temperature may help for Sokoban (precise task). But DAPO/JustRL show that temp=1.0 works well even for math.

#### 6.2.5 Token Budget

| Experiment | max_tokens | seq_len | Rationale |
|---|---|---|---|
| 8k (baseline) | 8192 | 16884 | Current config |
| 16k | 16384 | 32768 | GEPA showed truncation is a major bottleneck |
| progressive | 8K -> 16K | staged | DeepSeek-R1 pattern |

**Hypothesis**: Increasing max_tokens is the single highest-leverage change. Many rollouts get truncated at 8192 before producing `<answer>` tags. This wastes training signal.

#### 6.2.6 Importance Ratio Masking

| Experiment | token_mask_low | token_mask_high | Rationale |
|---|---|---|---|
| tight (baseline) | 0.125 | 8.0 | Current config |
| dapo_style | 0.2 | 1.28 | DAPO/JustRL asymmetric clipping range |
| loose | 0.05 | 20.0 | Very permissive |

**Hypothesis**: The current masking range (0.125-8.0) is much wider than DAPO's clipping (0.8-1.28). Since distillation is more on-policy than RL (the signal is denser), tighter masking may be safe and beneficial.

#### 6.2.7 Group Size (Rollouts per Example)

| Experiment | rollouts_per_example | batch_size | Rationale |
|---|---|---|---|
| 8_rollouts | 8 | 256 | JustRL uses 8 |
| 16_rollouts (baseline) | 16 | 256 | Current config, SkyRL uses 16 |
| 32_rollouts | 32 | 256 | More variance reduction |

**Hypothesis**: 16 rollouts is good. 8 is the minimum for stable advantage estimation. 32 may help at the cost of throughput.

### 6.3 Phase 3: GEPA-Optimized Teacher Prompts

**Goal**: Use GEPA to find teacher prompts that are optimal for distillation (not just for direct performance).

#### 6.3.1 GEPA Distillation Loop

1. Run GEPA to generate a Pareto front of teacher prompts (using the fixed procedure from Run 7)
2. For each Pareto front program, run context distillation for N steps
3. Evaluate the distilled student (without teacher context)
4. Select the teacher prompt that produces the best distilled student
5. (Optional) Use the distilled student as the new base model and repeat

#### 6.3.2 GEPA Settings for Distillation

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Reflection LM | Claude Opus | Best quality for prompt generation |
| Train examples | 500 | Balance speed vs coverage |
| Val examples | 200 | Enough for signal |
| Minibatch | 10 | Larger than Run 7 (was 5) for less noise |
| Temperature | 1.0 | Match RL training |
| Max tokens | 8192 | Match RL training |
| Rollout budget | 20K+ | More than Run 7's 10K |
| Cache | Clear before each run | Critical |
| Adapter | PrimeRLAdapter (field-name based) | Fixed version |

#### 6.3.3 Distillation Evaluation Protocol

For each GEPA-proposed teacher prompt:
1. Run 200 steps of context distillation with that prompt as teacher context
2. Evaluate the distilled student on 500 held-out examples (no teacher context)
3. Compare to baseline student (no distillation)
4. Track: final eval accuracy, teacher_kl curve, reward curve

This is expensive (~200 steps * N prompts) but gives the most accurate signal for which prompts produce the best distillation outcomes.

### 6.4 Phase 4: Stronger Teacher Models

**Goal**: Use a larger/better model as teacher for more meaningful distillation gains.

| Experiment | Teacher | Student | Config |
|---|---|---|---|
| same_model (baseline) | Qwen3-4B + context | Qwen3-4B | Current |
| larger_teacher | Qwen3-30B-A3B + context | Qwen3-4B | rl.toml has this commented out |
| rl_teacher | RL-trained Qwen3-4B | Qwen3-4B (fresh) | Distill RL capability into base |

The `rl.toml` already has the Qwen3-30B-A3B teacher config commented out:
```toml
# [teacher_inference.model]
# name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
# max_model_len = 8192
```

**Expected gains**: 5-15% absolute improvement with larger teacher (based on GKD and Thinking Machines results). The SkyRL script uses an RL-trained 4B as teacher for a 1.7B student, achieving significant gains.

### 6.5 Phase 5: Iterative Loop

If Phase 4 shows meaningful gains:
1. Take the best distilled student from Phase 4
2. Use it as the new student in a second round of distillation
3. Optionally, use the distilled student + RL (hybrid objective) for further improvement
4. Repeat 2-3 times, checking for diminishing returns

This mirrors the Qwen3 pipeline: SFT -> Distillation -> RL -> Repeat.

---

## 7. Recommended Configs

### 7.1 Pure Context Distillation (Baseline)

```toml
# experiments/context_distill/rl.toml (current config)
max_steps = 400
seq_len = 16884

[model]
name = "Qwen/Qwen3-4B-Instruct-2507"

[trainer.optim]
lr = 1e-5
betas1 = 0.85
betas2 = 0.9
weight_decay = 0.0
max_norm = 1.0

[trainer.model.lora]
rank = 16
alpha = 64

[trainer.loss]
adv_tau = 0.0
teacher_tau = 1.0
student_tau = 1.0

[orchestrator]
batch_size = 256
rollouts_per_example = 16

[orchestrator.advantage]
use_full_reward_baseline = true

[orchestrator.sampling]
max_tokens = 8192
temperature = 1.0
```

### 7.2 Hybrid: Task Reward + Distillation (Recommended First Ablation)

```toml
# Same as baseline except:
[trainer.loss]
adv_tau = 0.5
teacher_tau = 0.5
student_tau = 0.5
```

### 7.3 Extended Token Budget (Highest-Leverage Change)

```toml
# Same as baseline except:
seq_len = 32768

[orchestrator.sampling]
max_tokens = 16384
```

### 7.4 Larger Teacher Model

```toml
# Same as baseline except uncomment:
[teacher_inference.model]
name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
max_model_len = 8192

[teacher_inference.server]
port = 8001
```

### 7.5 Batch Job Template

```bash
#!/bin/bash
set -e
cd /global/homes/s/siddart2
export HOME=$SCRATCH
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
cd $SCRATCH

podman-hpc run --rm -it \
  --user "$(id -u):$(id -g)" --replace --name skyrl \
  --group-add keep-groups --userns keep-id --gpu --nccl --shm-size=8g \
  -e SCRATCH -e HOME \
  -e WANDB_API_KEY=595199cad0de28f309ce22cb212dcbeeb21b06d8 \
  -v "$SCRATCH":"$SCRATCH" -v "$HOME":"$HOME" \
  -w "$SCRATCH/prime-rl" \
  docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 \
  bash -c '
    unset NCCL_SOCKET_IFNAME
    source .venv/bin/activate

    uv run rl @ experiments/context_distill/rl.toml \
      --inference_gpu_ids 0 \
      --teacher_gpu_ids 1 \
      --trainer_gpu_ids 2,3 \
      --output_dir outputs/context_distill_baseline \
      --wandb.name context_distill_baseline
  '
```

Submit with:
```bash
sbatch -A m4881 -C "gpu&hbm80g" --qos=premium --time 24:00:00 --gpus-per-node 4 --output=logs/job_%j.out script.sh
```

---

## 8. References

### Papers

1. **DeepSeek-R1**: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (arXiv:2501.12948, Jan 2025)
2. **DeepSeek-Math**: "DeepSeek-Math: Pushing the Limits of Mathematical Reasoning in Open Language Models" (arXiv:2402.03300, Feb 2024)
3. **DAPO**: "DAPO: An Open-Source LLM Reinforcement Learning System" (arXiv:2503.14476, Mar 2025)
4. **Qwen3**: "Qwen3 Technical Report" (arXiv:2505.09388, May 2025)
5. **Llama 3.1**: "The Llama 3 Herd of Models" (arXiv:2407.21783, Jul 2024)
6. **GKD**: "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes" (arXiv:2306.13649, Jun 2023)
7. **MiniLLM**: "Knowledge Distillation of Large Language Models" (arXiv:2306.08543, Jun 2023)
8. **OPSD**: "On-Policy Self-Distillation for LLM Reasoning" (2025)
9. **PRIME**: "PRIME: Process Reinforcement through Implicit Rewards" (arXiv:2502.01456, 2025)
10. **Constitutional AI**: Bai et al. (2022) -- introduced context distillation concept

### Blog Posts and Reports

11. **JustRL**: "Just Use RL" blog (2025) -- simple GRPO recipe matching DeepSeek-R1
12. **Thinking Machines**: "On-Policy Distillation" blog -- 10x cheaper than RL, 74.4% AIME'24
13. **Sebastian Raschka**: "State of LLM Reasoning Training" (2025)
14. **CMU Blog**: "Exploration in LLM RL Training" (2025)
15. **NovaSky/SkyRL**: On-policy distillation notion page and codebase

### Codebases

16. **prime-rl**: `/pscratch/sd/s/siddart2/prime-rl/` -- main training framework
17. **SkyRL**: `/pscratch/sd/s/siddart2/SkyRL/` -- reference on-policy distillation implementation
18. **dspy_gepa**: `/pscratch/sd/s/siddart2/contextual_learning/dspy_gepa/` -- GEPA prompt optimization

---

*Document created: Feb 7, 2026*
*Based on: literature survey, codebase analysis, GEPA Run 7 results, code review of prime-rl distillation pipeline*
