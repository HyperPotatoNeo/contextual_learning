#!/usr/bin/env python3
"""
GEPA Prompt Optimization for Sokoban Puzzle Solving

This script implements GEPA (Genetic-Pareto) prompt optimization from DSPy
for the Sokoban puzzle-solving environment. It uses vLLM inference with
configurable student and reflection models.

GEPA is a reflective prompt optimizer that:
- Uses LLMs to reflect on program execution traces
- Identifies what worked and what didn't
- Proposes improved prompts based on feedback
- Can outperform RL methods with fewer rollouts

Features:
- Uses system prompt from verifiers (reasoning-gym) environment
- Scoring directly from reasoning_gym's Sokoban verifier
- Configurable student model (for solving) and reflection model (for optimization)
- Support for vLLM with tensor parallelism across multiple GPUs
- Proper train/val/test splits for robust evaluation

Usage:
    # Basic run
    python sokoban_gepa.py

    # Custom models
    python sokoban_gepa.py --model Qwen/Qwen3-4B-Instruct-2507 --reflection-model meta-llama/Llama-3.1-8B-Instruct

    # Custom dataset sizes
    python sokoban_gepa.py --train-size 200 --val-size 100 --test-size 200

    # Dry run to test setup
    python sokoban_gepa.py --dry-run
"""

import argparse
import random
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import dspy

# Import reasoning_gym directly (avoids verifiers CompositeDataset indirection)
import reasoning_gym as rg
from dspy.adapters.chat_adapter import ChatAdapter
from rich.console import Console
from rich.table import Table

console = Console()


# =============================================================================
# Custom Adapter: Match prime-rl prompt format exactly
# =============================================================================


class PrimeRLAdapter(ChatAdapter):
    """
    Custom DSPy adapter that sends messages in the exact format used by prime-rl
    for context distillation — but ONLY for the SokobanSolver signature.

    For SokobanSolver (student model):
        System: {signature.instructions}  (the teacher context / optimized prompt)
        User: {question}  (the raw sokoban puzzle question)

    For all other signatures (e.g., GEPA reflection with Claude Opus):
        Falls back to default ChatAdapter behavior with field markers.

    This ensures:
    1. The student model sees the exact same format as prime-rl during RL training
    2. GEPA's reflection model still gets proper structured input/output

    IMPORTANT: Must be set as global adapter via dspy.configure(adapter=PrimeRLAdapter())
    Setting lm.adapter does NOT work — DSPy ignores it.
    """

    # Identify raw-format signatures by their field structure, NOT by __name__.
    # GEPA's build_program() calls signature.with_instructions() which creates a new
    # signature with __name__="StringSignature" instead of "SokobanSolver".
    # Checking field names is robust to this transformation.
    RAW_FORMAT_FIELDS = (frozenset({"question"}), frozenset({"answer"}))

    def _is_raw_format_signature(self, signature):
        """Check if this signature should use raw prime-rl format.

        Uses field names instead of __name__ because GEPA's with_instructions()
        creates StringSignature objects that lose the original class name.
        """
        input_keys = frozenset(signature.input_fields.keys())
        output_keys = frozenset(signature.output_fields.keys())
        return (input_keys, output_keys) == self.RAW_FORMAT_FIELDS

    def format(self, signature, demos, inputs):
        """Format messages: raw for SokobanSolver, ChatAdapter for everything else."""
        if self._is_raw_format_signature(signature):
            messages = []
            instruction = signature.instructions
            if instruction:
                messages.append({"role": "system", "content": instruction})
            question = inputs.get("question", "")
            messages.append({"role": "user", "content": question})
            return messages
        # Fall back to ChatAdapter for reflection and other signatures
        return super().format(signature, demos, inputs)

    def parse(self, signature, completion):
        """Parse output: raw for SokobanSolver, ChatAdapter for everything else."""
        if self._is_raw_format_signature(signature):
            return {"answer": completion}
        return super().parse(signature, completion)


# =============================================================================
# Configuration
# =============================================================================

# Default Sokoban configuration matching rl.toml
DEFAULT_CONFIG = {
    "seed": 42,
    "min_w": 4,
    "max_w": 9,
    "min_h": 4,
    "max_h": 9,
    "min_boxes": 1,
    "max_boxes": 7,
    "max_depth": 80,
}

# Default system prompt from verifiers (reasoning-gym)
# This is used as the docstring for the DSPy Signature
VERIFIERS_SYSTEM_PROMPT = """Given a problem, your task is to answer the question by thinking step-by-step in a clear and specific manner.
Once you have thought about the reasoning process, provide the answer in the following format:
<answer>answer here</answer>
Do not explain your reasoning inside the answer tags, provide only the final answer. When an example is provided, you should strictly follow the format of the output/answer in that example."""

# Default teacher prompt file path (relative to this script)
DEFAULT_TEACHER_PROMPT_FILE = str(Path(__file__).parent / "teacher_prompt.txt")


# =============================================================================
# Data Loading
# =============================================================================


def load_datasets(config: dict, train_size: int = 100, val_size: int = 50, test_size: int = 100):
    """
    Load and split datasets for GEPA optimization.

    Uses rg.create_dataset() directly (not the verifiers CompositeDataset wrapper)
    to get puzzles from indices 0..N, matching standalone eval behavior.

    The previous approach used verifiers.ReasoningGymEnv which created a 5000+N item
    dataset and used indices 5000..5000+N for eval, producing different (harder) puzzles.

    Args:
        config: Sokoban environment configuration
        train_size: Number of training examples
        val_size: Number of validation examples
        test_size: Number of test examples (held-out)

    Returns:
        train_set: List of dspy.Example for training
        val_set: List of dspy.Example for validation
        test_set: List of dspy.Example for held-out testing
        system_prompt: The system prompt string (VERIFIERS_SYSTEM_PROMPT)
    """
    console.print("[bold blue]Loading Sokoban datasets...[/bold blue]")

    sokoban_config = {
        "min_w": config["min_w"],
        "max_w": config["max_w"],
        "min_h": config["min_h"],
        "max_h": config["max_h"],
        "min_boxes": config["min_boxes"],
        "max_boxes": config["max_boxes"],
        "max_depth": config["max_depth"],
    }

    # Create train+val dataset with one seed, test with different seed
    train_val_rg = rg.create_dataset("sokoban", size=train_size + val_size, seed=config["seed"], **sokoban_config)
    test_rg = rg.create_dataset("sokoban", size=test_size, seed=config["seed"] + 1000, **sokoban_config)

    # Convert to list of dicts
    train_val_ds = [train_val_rg[i] for i in range(len(train_val_rg))]
    test_ds = [test_rg[i] for i in range(len(test_rg))]

    # Shuffle and split train/val
    random.Random(config["seed"]).shuffle(train_val_ds)

    def convert_to_dspy_example(ex: dict) -> dspy.Example:
        """Convert reasoning_gym format to DSPy Example."""
        return dspy.Example(
            {
                "question": ex["question"],
                "answer": ex.get("answer", ""),
            }
        ).with_inputs("question")

    train_set = [convert_to_dspy_example(ex) for ex in train_val_ds[:train_size]]
    val_set = [convert_to_dspy_example(ex) for ex in train_val_ds[train_size : train_size + val_size]]
    test_set = [convert_to_dspy_example(ex) for ex in test_ds]

    console.print(f"  Train set: {len(train_set)} examples")
    console.print(f"  Val set: {len(val_set)} examples")
    console.print(f"  Test set: {len(test_set)} examples")
    console.print("  Dataset: direct rg.create_dataset (indices 0..N, same as standalone eval)")

    return train_set, val_set, test_set, VERIFIERS_SYSTEM_PROMPT


# =============================================================================
# DSPy Signatures and Programs
# =============================================================================


class SokobanSolver(dspy.Signature):
    """Given a problem, your task is to answer the question by thinking step-by-step in a clear and specific manner.
    Once you have thought about the reasoning process, provide the answer in the following format:
    <answer>answer here</answer>
    Do not explain your reasoning inside the answer tags, provide only the final answer. When an example is provided, you should strictly follow the format of the output/answer in that example."""

    question = dspy.InputField(desc="The Sokoban puzzle description including the grid layout")
    answer = dspy.OutputField(desc="The solution moves wrapped in <answer> tags, e.g., <answer>LRUD</answer>")


class SokobanProgram(dspy.Module):
    """
    DSPy program for solving Sokoban puzzles.

    This uses a simple Predict module (not ChainOfThought) to keep outputs concise
    and avoid token limit issues.
    """

    def __init__(self):
        super().__init__()
        self.solver = dspy.Predict(SokobanSolver)

    def forward(self, question: str):
        return self.solver(question=question)


# =============================================================================
# Scoring / Metric
# =============================================================================


def extract_moves_from_response(response: str) -> tuple[str, str]:
    """
    Extract move sequence from model response.

    Handles various formats:
    - <answer>LRUD</answer> XML tags
    - Truncated <answer>LRUD (no closing tag)
    - Raw move sequences like "LRUD"
    - Moves embedded in text

    Args:
        response: Raw model response

    Returns:
        Tuple of (extracted_moves, extraction_note) where extraction_note
        describes how the moves were found (for feedback to reflection model)
    """
    if not response:
        return "", "empty_response"

    response = response.strip()

    # Try to extract from complete <answer>...</answer> tags first
    if "<answer>" in response and "</answer>" in response:
        start = response.find("<answer>") + 8
        end = response.find("</answer>")
        tag_content = response[start:end].strip()
        moves_pattern = r"[LRUD]+"
        matches = re.findall(moves_pattern, tag_content.upper())
        if matches:
            return max(matches, key=len), "answer_tags"
        return "", "answer_tags_empty"

    # Handle truncated <answer> tag (opening tag but no closing - likely truncated)
    if "<answer>" in response:
        start = response.find("<answer>") + 8
        tag_content = response[start:].strip()
        moves_pattern = r"[LRUD]+"
        matches = re.findall(moves_pattern, tag_content.upper())
        if matches:
            return max(matches, key=len), "truncated_answer_tag"
        return "", "truncated_answer_tag_empty"

    # No <answer> tags at all - extract the longest sequence of valid moves from full text
    moves_pattern = r"[LRUD]+"
    matches = re.findall(moves_pattern, response.upper())
    if matches:
        return max(matches, key=len), "no_answer_tags"

    return "", "no_moves_found"


def create_metric():
    """
    Create a metric function for GEPA optimization.

    Uses reasoning_gym's SokobanDataset.score_answer() for verification,
    which simulates the moves on the puzzle grid and checks if all boxes
    are on goal positions.

    Returns:
        Metric function compatible with DSPy GEPA
    """
    from reasoning_gym.games.sokoban import SokobanConfig, SokobanDataset

    # Create a reusable dataset instance for scoring
    config = SokobanConfig(size=1, seed=0)
    sokoban_scorer = SokobanDataset(config)

    def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):  # noqa: ARG001
        """
        Score a prediction using reasoning_gym's Sokoban verifier.

        Provides detailed feedback for the reflection model about what went wrong,
        including truncation detection, missing answer tags, and empty responses.

        Args:
            example: DSPy Example with 'question' and 'answer' fields
            prediction: Model prediction with 'answer' field
            trace: Optional execution trace
            pred_name: Optional prediction name
            pred_trace: Optional prediction trace

        Returns:
            dspy.Prediction with score (0.0 or 1.0) and feedback
        """
        # With PrimeRLAdapter, prediction.answer contains the full raw model output
        # (including <think>...</think> reasoning tokens from Qwen3)
        has_answer_field = hasattr(prediction, "answer") and prediction.answer
        has_reasoning_field = hasattr(prediction, "reasoning") and prediction.reasoning

        if has_answer_field:
            raw_answer = prediction.answer
        elif has_reasoning_field:
            raw_answer = prediction.reasoning
        else:
            raw_answer = str(prediction) if prediction else ""

        # Detect truncation: if response doesn't contain <answer> tags, it may be truncated
        response_truncated = "<answer>" not in raw_answer if raw_answer else True

        # Extract moves from response (now returns a tuple with extraction notes)
        predicted_moves, extraction_note = extract_moves_from_response(raw_answer)

        # Build detailed feedback about extraction issues
        extraction_issues = []
        if response_truncated:
            extraction_issues.append(
                "CRITICAL: The model's response was TRUNCATED before it could produce a final answer. "
                "The response ran out of tokens. The prompt should instruct the model to be more concise "
                "in its reasoning and produce the answer earlier, or to output the answer FIRST before "
                "showing detailed work."
            )
        if extraction_note == "empty_response":
            extraction_issues.append("The model produced an EMPTY response with no content at all.")
        elif extraction_note == "answer_tags_empty":
            extraction_issues.append(
                "The model used <answer>...</answer> tags but they were EMPTY - no moves inside the tags."
            )
        elif extraction_note == "truncated_answer_tag":
            extraction_issues.append(
                "The model started writing <answer> but was TRUNCATED before </answer>. "
                "The response ran out of tokens mid-answer."
            )
        elif extraction_note == "truncated_answer_tag_empty":
            extraction_issues.append(
                "The model started writing <answer> but was TRUNCATED and no valid moves were found after the tag."
            )
        elif extraction_note == "no_answer_tags":
            extraction_issues.append(
                "The model did NOT use the required <answer>...</answer> format. "
                "Moves were extracted from the raw text, which may be unreliable. "
                "The prompt should emphasize using <answer>MOVES</answer> format."
            )
        elif extraction_note == "no_moves_found":
            extraction_issues.append(
                "NO valid moves (L, R, U, D) were found anywhere in the model's response. "
                "The model completely failed to produce a move sequence."
            )

        # Extract gamestr from question for scoring
        question_str = example.question
        lines = question_str.split("\n")

        puzzle_start = None
        for i, line in enumerate(lines):
            if "Here is your puzzle:" in line:
                puzzle_start = i + 1
                break

        if puzzle_start is None:
            return dspy.Prediction(score=0.0, feedback="Could not find puzzle grid in question.")

        gamestr_lines = []
        for line in lines[puzzle_start:]:
            if line.strip():
                gamestr_lines.append(line)
            else:
                break
        gamestr = "\n".join(gamestr_lines)

        # If no moves were extracted, return early with detailed feedback
        if not predicted_moves:
            issue_text = " ".join(extraction_issues) if extraction_issues else "No moves were produced."
            return dspy.Prediction(score=0.0, feedback=f"FAILED: No valid move sequence was produced. {issue_text}")

        # Score using reasoning_gym
        entry = {"metadata": {"gamestr": gamestr}}
        try:
            score = sokoban_scorer.score_answer(predicted_moves, entry)
        except Exception as e:
            issue_text = " ".join(extraction_issues) if extraction_issues else ""
            return dspy.Prediction(
                score=0.0, feedback=f"Error scoring answer: {e}. Extracted moves: '{predicted_moves}'. {issue_text}"
            )

        if score == 1.0:
            return dspy.Prediction(
                score=1.0,
                feedback=f"Correct! The solution '{predicted_moves}' ({len(predicted_moves)} moves) solves the puzzle.",
            )
        else:
            issue_text = " ".join(extraction_issues) if extraction_issues else ""
            return dspy.Prediction(
                score=0.0,
                feedback=f"Incorrect. The extracted moves '{predicted_moves}' ({len(predicted_moves)} moves) "
                f"do not solve the puzzle. The moves were applied but not all boxes ended up on goals. "
                f"{issue_text}",
            )

    return metric


# =============================================================================
# Model Setup
# =============================================================================


def setup_vllm_lm(
    model_name: str,
    api_base: str = "http://localhost:8000/v1",
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> dspy.LM:
    """
    Set up a vLLM-based language model for DSPy.

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-4B-Instruct-2507")
        api_base: vLLM server API endpoint
        max_tokens: Maximum generation tokens
        temperature: Sampling temperature

    Returns:
        Configured dspy.LM instance
    """
    console.print(f"[bold blue]Setting up LM: {model_name}[/bold blue]")
    console.print(f"  API base: {api_base}")
    console.print(f"  Max tokens: {max_tokens}")
    console.print(f"  Temperature: {temperature}")

    lm = dspy.LM(
        model=f"openai/{model_name}",
        api_base=api_base,
        api_key="dummy",  # vLLM doesn't need real key
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return lm


def setup_anthropic_lm(
    model_name: str = "claude-sonnet-4-20250514",
    api_key: str = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> dspy.LM:
    """
    Set up an Anthropic Claude model for DSPy (used as reflection model).

    Args:
        model_name: Anthropic model name (e.g., "claude-sonnet-4-20250514")
        api_key: Anthropic API key
        max_tokens: Maximum generation tokens
        temperature: Sampling temperature

    Returns:
        Configured dspy.LM instance
    """
    console.print(f"[bold blue]Setting up Anthropic LM: {model_name}[/bold blue]")
    console.print(f"  Max tokens: {max_tokens}")
    console.print(f"  Temperature: {temperature}")

    lm = dspy.LM(
        model=f"anthropic/{model_name}",
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return lm


# =============================================================================
# Evaluation
# =============================================================================


def run_evaluation(
    program: dspy.Module,
    dataset: list,
    metric,
    num_threads: int = 32,
    label: str = "Evaluation",
) -> float:
    """
    Run evaluation on a dataset.

    Args:
        program: DSPy program to evaluate
        dataset: List of dspy.Example
        metric: Scoring function
        num_threads: Parallelism for evaluation
        label: Label for console output

    Returns:
        Accuracy score (0.0 to 1.0)
    """
    console.print(f"[bold blue]Running {label}...[/bold blue]")

    evaluate = dspy.Evaluate(
        devset=dataset,
        metric=metric,
        num_threads=num_threads,
        display_progress=True,
    )

    result = evaluate(program)

    # Extract score from EvaluationResult
    if hasattr(result, "score"):
        score = result.score
    elif hasattr(result, "__float__"):
        score = float(result)
    else:
        score = result

    # Normalize score to 0-1 range if needed
    # DSPy Evaluate sometimes returns percentage (0-100) or fraction (0-1)
    if isinstance(score, (int, float)):
        if score > 1:
            # Score is a percentage, convert to fraction
            score = score / 100.0

    # Display score
    if isinstance(score, (int, float)):
        console.print(f"[bold green]{label} accuracy: {score:.2%}[/bold green]")
    else:
        console.print(f"[bold green]{label} accuracy: {score}[/bold green]")

    return score


# =============================================================================
# GEPA Optimization
# =============================================================================


def run_gepa_optimization(
    program: dspy.Module,
    train_set: list,
    val_set: list,
    metric,
    reflection_lm: dspy.LM,
    max_metric_calls: int = 1000,
    num_threads: int = 32,
    log_dir: str = None,
    reflection_minibatch_size: int = 5,
) -> dspy.Module:
    """
    Run GEPA (Genetic-Pareto) prompt optimization.

    GEPA uses the reflection_lm to analyze execution traces and propose
    improved prompts. It maintains a Pareto frontier of solutions.

    Args:
        program: DSPy program to optimize
        train_set: Training examples for optimization
        val_set: Validation examples for selection
        metric: Scoring function (returns score and feedback)
        reflection_lm: LM to use for reflection/optimization
        max_metric_calls: Budget for metric evaluations
        num_threads: Parallelism for evaluation
        log_dir: Directory for saving optimization logs
        reflection_minibatch_size: Number of examples per reflection step (default 5)

    Returns:
        Optimized DSPy program
    """
    console.print("[bold blue]Starting GEPA optimization...[/bold blue]")
    console.print(f"  Max metric calls: {max_metric_calls}")
    console.print(f"  Train size: {len(train_set)}")
    console.print(f"  Val size: {len(val_set)}")
    console.print(f"  Reflection LM: {reflection_lm.model}")
    console.print(f"  Reflection minibatch size: {reflection_minibatch_size}")

    optimizer = dspy.GEPA(
        metric=metric,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
        num_threads=num_threads,
        track_stats=True,
        log_dir=log_dir,
        reflection_minibatch_size=reflection_minibatch_size,
        add_format_failure_as_feedback=True,
    )

    optimized = optimizer.compile(
        student=program,
        trainset=train_set,
        valset=val_set,
    )

    return optimized


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="GEPA Prompt Optimization for Sokoban",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with defaults
  python sokoban_gepa.py

  # Use different student and reflection models
  python sokoban_gepa.py --model Qwen/Qwen3-4B-Instruct-2507 \\
                          --reflection-model meta-llama/Llama-3.1-8B-Instruct

  # Custom dataset sizes
  python sokoban_gepa.py --train-size 200 --val-size 100 --test-size 200

  # Quick test
  python sokoban_gepa.py --dry-run

  # Baseline evaluation only
  python sokoban_gepa.py --baseline-only

  # Use different vLLM servers for student and reflection
  python sokoban_gepa.py --api-base http://localhost:8000/v1 \\
                          --reflection-api-base http://localhost:8001/v1
        """,
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Student model name for solving puzzles (default: Qwen/Qwen3-4B-Instruct-2507)",
    )
    parser.add_argument(
        "--reflection-model",
        type=str,
        default=None,
        help="Reflection model for GEPA optimization. Use 'claude' for Claude Sonnet, "
        "or a full model name like 'claude-sonnet-4-20250514'. (default: same as --model)",
    )
    parser.add_argument(
        "--api-base", type=str, default="http://localhost:8000/v1", help="vLLM API base URL for student model"
    )
    parser.add_argument(
        "--reflection-api-base",
        type=str,
        default=None,
        help="vLLM API base URL for reflection model (default: same as --api-base)",
    )
    parser.add_argument(
        "--anthropic-api-key",
        type=str,
        default=None,
        help="Anthropic API key for Claude reflection model",
    )
    parser.add_argument("--max-tokens", type=int, default=8192, help="Maximum generation tokens for student model")
    parser.add_argument(
        "--reflection-max-tokens",
        type=int,
        default=16384,
        help="Maximum generation tokens for reflection model (default: 16384, uncapped from previous 8192)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0, matching rl.toml)"
    )

    # Dataset configuration
    parser.add_argument("--train-size", type=int, default=500, help="Training set size (default: 500)")
    parser.add_argument("--val-size", type=int, default=150, help="Validation set size (default: 150)")
    parser.add_argument("--test-size", type=int, default=1000, help="Test set size (held-out, default: 1000)")

    # GEPA configuration
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=10000,
        help="GEPA optimization budget (metric evaluations, default: 10000)",
    )
    parser.add_argument(
        "--reflection-minibatch-size",
        type=int,
        default=20,
        help="Number of examples per GEPA reflection step (default: 20)",
    )
    parser.add_argument("--num-threads", type=int, default=32, help="Parallelism for evaluation")

    # Output configuration
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")

    # Run modes
    parser.add_argument("--dry-run", action="store_true", help="Just load data and show config, don't run optimization")
    parser.add_argument("--baseline-only", action="store_true", help="Only run baseline evaluation, skip optimization")
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="Resume from existing output directory (with gepa_logs/gepa_state.bin)",
    )
    parser.add_argument(
        "--teacher-prompt-file",
        type=str,
        default=None,
        help="Path to a text file containing the teacher prompt to use as GEPA starting point. "
        "Defaults to teacher_prompt.txt in the same directory as this script.",
    )

    args = parser.parse_args()

    # Create or resume output directory
    if args.resume_dir:
        output_dir = Path(args.resume_dir)
        if not (output_dir / "gepa_logs" / "gepa_state.bin").exists():
            console.print(f"[red]Error: No gepa_state.bin found in {output_dir / 'gepa_logs'}[/red]")
            sys.exit(1)
        console.print(f"[green]Resuming from: {output_dir}[/green]")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / f"gepa_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold]GEPA Prompt Optimization for Sokoban[/bold]")
    console.print(f"Output directory: {output_dir}\n")

    # NOTE: DSPy disk cache must be cleared BEFORE this script runs (see run script).
    # DSPy caches LM responses on disk (even at temperature>0) and returns the same
    # stochastic result for identical requests. If a previous run cached bad results
    # for the same prompts, they would be returned instead of fresh API calls.
    # This caused Run 5's baseline to show 12% instead of the true 40%.
    # The cache is at $HOME/.dspy_cache and must be cleared in the shell script
    # before Python starts, since DSPy opens the database on import.

    # Load datasets
    train_set, val_set, test_set, _system_prompt = load_datasets(
        DEFAULT_CONFIG,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
    )

    if args.dry_run:
        console.print("\n[yellow]Dry run - exiting without optimization[/yellow]")
        console.print("\nConfiguration:")
        console.print(f"  Student model: {args.model}")
        console.print(f"  Reflection model: {args.reflection_model or args.model}")
        console.print(f"  API base: {args.api_base}")
        console.print(f"  Max tokens: {args.max_tokens}")
        console.print(f"  Train/Val/Test: {args.train_size}/{args.val_size}/{args.test_size}")
        console.print(f"  Max metric calls: {args.max_metric_calls}")
        return

    # Set up language models
    student_lm = setup_vllm_lm(
        args.model,
        api_base=args.api_base,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    # CRITICAL: Set PrimeRLAdapter as the GLOBAL adapter via dspy.configure().
    # Setting lm.adapter does NOT work — DSPy's Predict.forward() uses settings.adapter,
    # not lm.adapter. PrimeRLAdapter is signature-aware: it uses raw format only for
    # SokobanSolver and falls back to ChatAdapter for GEPA reflection signatures.
    adapter = PrimeRLAdapter()
    dspy.configure(lm=student_lm, adapter=adapter)
    console.print("  [bold cyan]Using PrimeRLAdapter (global, signature-aware)[/bold cyan]")
    console.print("  [bold cyan]  SokobanSolver → raw system + user messages[/bold cyan]")
    console.print("  [bold cyan]  Other signatures → ChatAdapter (field markers)[/bold cyan]")

    # Set up reflection model (can be different from student)
    reflection_model = args.reflection_model or args.model

    # Check if reflection model is Claude (Anthropic)
    if reflection_model and reflection_model.startswith("claude"):
        # Map shorthand to full model names
        claude_models = {
            "claude": "claude-sonnet-4-20250514",
            "claude-sonnet": "claude-sonnet-4-20250514",
            "claude-opus": "claude-opus-4-6",
            "claude-haiku": "claude-haiku-4-5-20251001",
        }
        reflection_model = claude_models.get(reflection_model, reflection_model)
        reflection_lm = setup_anthropic_lm(
            model_name=reflection_model,
            api_key=args.anthropic_api_key,
            max_tokens=args.reflection_max_tokens,
            temperature=args.temperature,
        )
    else:
        reflection_api_base = args.reflection_api_base or args.api_base
        reflection_lm = setup_vllm_lm(
            reflection_model,
            api_base=reflection_api_base,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    # Create program and metric
    program = SokobanProgram()
    metric = create_metric()

    # Override starting prompt if --teacher-prompt-file is set
    if args.teacher_prompt_file:
        prompt_file = Path(args.teacher_prompt_file)
        if not prompt_file.exists():
            console.print(f"[red]Error: Teacher prompt file not found: {prompt_file}[/red]")
            sys.exit(1)
        teacher_prompt = prompt_file.read_text().strip()
        program.solver.signature = program.solver.signature.with_instructions(teacher_prompt)
        console.print(f"  [bold green]Using teacher prompt from: {prompt_file}[/bold green]")
        console.print(f"  [dim]{len(teacher_prompt)} chars, {teacher_prompt.count(chr(10)) + 1} lines[/dim]")

    # Run baseline evaluation on full test set
    baseline_score = run_evaluation(program, test_set, metric, args.num_threads, "Baseline")

    if args.baseline_only:
        console.print("\n[yellow]Baseline only - exiting[/yellow]")
        return

    # Run GEPA optimization
    optimized = run_gepa_optimization(
        program=program,
        train_set=train_set,
        val_set=val_set,
        metric=metric,
        reflection_lm=reflection_lm,
        max_metric_calls=args.max_metric_calls,
        num_threads=args.num_threads,
        log_dir=str(output_dir / "gepa_logs"),
        reflection_minibatch_size=args.reflection_minibatch_size,
    )

    # Evaluate optimized program on full test set
    final_score = run_evaluation(optimized, test_set, metric, args.num_threads, "Optimized (Test Set)")

    # Print results summary
    console.print("\n[bold]Results Summary[/bold]")
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    if isinstance(baseline_score, (int, float)) and isinstance(final_score, (int, float)):
        table.add_row("Baseline Accuracy", f"{baseline_score:.2%}")
        table.add_row("Optimized Accuracy", f"{final_score:.2%}")
        table.add_row("Improvement", f"{(final_score - baseline_score):+.2%}")
    else:
        table.add_row("Baseline Accuracy", str(baseline_score))
        table.add_row("Optimized Accuracy", str(final_score))

    table.add_row("Student Model", args.model)
    table.add_row("Reflection Model", reflection_model)
    console.print(table)

    # Save optimized program
    optimized_path = output_dir / "optimized_program.json"
    optimized.save(str(optimized_path))
    console.print(f"\nSaved optimized program to: {optimized_path}")

    # Save optimized system prompt (for context distillation in prime-rl)
    try:
        opt_instruction = optimized.solver.signature.instructions
        prompt_path = output_dir / "system_prompt.txt"
        with open(prompt_path, "w") as f:
            f.write(opt_instruction)
        console.print(f"Saved optimized system prompt to: {prompt_path}")
        console.print(f"  Prompt: {opt_instruction[:200]}...")
    except Exception as e:
        console.print(f"[yellow]Could not extract system prompt: {e}[/yellow]")

    # Save results summary
    results_path = output_dir / "results.txt"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(results_path, "w") as f:
        f.write("GEPA Optimization Results\n")
        f.write("========================\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Student Model: {args.model}\n")
        f.write(f"Reflection Model: {reflection_model}\n")
        f.write(f"Train/Val/Test: {args.train_size}/{args.val_size}/{args.test_size}\n")
        f.write(f"Max Metric Calls: {args.max_metric_calls}\n\n")
        f.write(f"Baseline Accuracy: {baseline_score}\n")
        f.write(f"Optimized Accuracy: {final_score}\n")
        if isinstance(baseline_score, (int, float)) and isinstance(final_score, (int, float)):
            f.write(f"Improvement: {final_score - baseline_score:+.4f}\n")
    console.print(f"Saved results to: {results_path}")


if __name__ == "__main__":
    main()
