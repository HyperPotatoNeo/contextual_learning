#!/usr/bin/env python3
"""
Standalone evaluation of the GEPA-optimized prompt, matching prime-rl exactly.

Tests multiple prompt application methods:
1. "prime_rl_teacher": System=GEPA_prompt, User=sokoban_question (how prime-rl baseline eval does it)
2. "prime_rl_student": System=default_reasoning_gym_prompt, User=sokoban_question (student baseline)
3. "dspy_style": System=DSPy_formatted(GEPA_prompt), User=[[ ## question ## ]]\nsokoban_question

Uses the exact same:
- Sokoban config as rl.toml
- Scoring function (reasoning_gym SokobanDataset.score_answer)
- vLLM inference
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from openai import AsyncOpenAI

# Import reasoning_gym for scoring
sys.path.insert(0, "/pscratch/sd/s/siddart2/dspy_gepa")

import reasoning_gym as rg
from reasoning_gym.utils import SYSTEM_PROMPTS

DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPTS["default"]

# Sokoban config matching rl.toml exactly
SOKOBAN_CONFIG = {
    "min_w": 4,
    "max_w": 9,
    "min_h": 4,
    "max_h": 9,
    "min_boxes": 1,
    "max_boxes": 7,
    "max_depth": 80,
}


def load_sokoban_dataset(num_examples: int, seed: int = 42):
    """Load Sokoban puzzles using reasoning_gym directly."""
    dataset = rg.create_dataset(
        "sokoban",
        size=num_examples,
        seed=seed,
        **SOKOBAN_CONFIG,
    )
    return dataset


def extract_answer_verifiers_style(completion: str) -> str | None:
    """Extract answer using the same regex as verifiers XMLParser."""
    regex = r"<answer>\s?(.*?)\s?</answer>"
    matches = list(re.finditer(regex, completion, flags=re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def extract_moves(text: str | None) -> str:
    """Extract valid move characters from text."""
    if not text:
        return ""
    return re.sub(r"[^LRUDlrud]", "", text).upper()


def format_dspy_system_message(instruction: str) -> str:
    """Format the system message exactly as DSPy ChatAdapter does."""
    import textwrap

    dedented = textwrap.dedent(instruction)
    objective = ("\n" + " " * 8).join([""] + dedented.splitlines())

    field_desc = (
        "Your input fields are:\n"
        "`question`: The Sokoban puzzle description including the grid layout\n\n"
        "Your output fields are:\n"
        "`answer`: The solution moves wrapped in <answer> tags, e.g., <answer>LRUD</answer>\n\n"
    )
    field_structure = (
        "All interactions will be structured in the following way, with the appropriate values filled in.\n\n"
        "[[ ## question ## ]]\n"
        "${question}\n\n"
        "[[ ## answer ## ]]\n"
        "${answer}\n\n"
        "[[ ## completed ## ]]\n\n"
    )
    task_desc = f"In adhering to this structure, your objective is: {objective}"

    return f"{field_desc}{field_structure}{task_desc}"


async def evaluate_single(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    dataset: rg.dataset.ProceduralDataset,
    idx: int,
    max_tokens: int = 8192,
    temperature: float = 1.0,
) -> dict:
    """Evaluate a single example."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        completion = response.choices[0].message.content or ""

        # Score using reasoning_gym exactly like verifiers does
        answer_text = extract_answer_verifiers_style(completion)
        moves = extract_moves(answer_text)
        entry = dataset[idx]
        score = dataset.score_answer(answer=moves, entry=entry)

        return {
            "idx": idx,
            "score": score,
            "moves": moves,
            "answer_text": answer_text,
            "completion_len": len(completion),
            "has_answer_tags": answer_text is not None,
        }
    except Exception as e:
        return {"idx": idx, "score": 0.0, "error": str(e)}


async def evaluate_method(
    client: AsyncOpenAI,
    model: str,
    dataset: rg.dataset.ProceduralDataset,
    method: str,
    gepa_prompt: str,
    num_examples: int,
    max_tokens: int = 8192,
    temperature: float = 1.0,
    concurrency: int = 32,
) -> list[dict]:
    """Evaluate a specific prompt method on the dataset."""
    sem = asyncio.Semaphore(concurrency)

    async def limited_eval(idx):
        async with sem:
            entry = dataset[idx]
            question = entry["question"]

            if method == "prime_rl_teacher":
                # How prime-rl baseline eval applies teacher context:
                # System = teacher_context (REPLACES default system prompt)
                # User = raw sokoban question
                messages = [
                    {"role": "system", "content": gepa_prompt},
                    {"role": "user", "content": question},
                ]
            elif method == "prime_rl_student":
                # How prime-rl baseline eval runs student (no context):
                # System = default reasoning_gym system prompt
                # User = raw sokoban question
                messages = [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ]
            elif method == "dspy_style":
                # How DSPy applies the prompt during GEPA evaluation:
                # System = DSPy formatted (field descriptions + structure + instruction)
                # User = "[[ ## question ## ]]\n{question}"
                messages = [
                    {"role": "system", "content": format_dspy_system_message(gepa_prompt)},
                    {"role": "user", "content": f"[[ ## question ## ]]\n{question}"},
                ]
            elif method == "combined":
                # Teacher context as system prompt + default system prompt merged
                combined = gepa_prompt + "\n\n" + DEFAULT_SYSTEM_PROMPT
                messages = [
                    {"role": "system", "content": combined},
                    {"role": "user", "content": question},
                ]
            else:
                raise ValueError(f"Unknown method: {method}")

            return await evaluate_single(
                client,
                model,
                messages,
                dataset,
                idx,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    tasks = [limited_eval(i) for i in range(num_examples)]
    results = await asyncio.gather(*tasks)
    return results


async def main():
    parser = argparse.ArgumentParser(description="Standalone prompt evaluation matching prime-rl")
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="outputs/gepa_20260207_041355/system_prompt.txt",
        help="Path to the GEPA-optimized prompt file",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--num-examples", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42, help="Seed for dataset generation (42 = eval set in rl.toml)")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["prime_rl_student", "prime_rl_teacher", "dspy_style"],
        help="Methods to evaluate",
    )
    args = parser.parse_args()

    # Load GEPA prompt
    prompt_path = Path(args.prompt_file)
    if not prompt_path.exists():
        print(f"Prompt file not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    gepa_prompt = prompt_path.read_text().strip()
    print(f"Loaded GEPA prompt: {len(gepa_prompt)} chars from {prompt_path}")

    # Load dataset
    print(f"Loading Sokoban dataset: {args.num_examples} examples, seed={args.seed}")
    print(f"Config: {SOKOBAN_CONFIG}")
    dataset = load_sokoban_dataset(args.num_examples, seed=args.seed)
    print(f"Dataset loaded: {len(dataset)} examples")

    # Create OpenAI client
    client = AsyncOpenAI(base_url=args.api_base, api_key="dummy")

    # Evaluate each method
    print(f"\nModel: {args.model}")
    print(f"Max tokens: {args.max_tokens}, Temperature: {args.temperature}")
    print(f"Concurrency: {args.concurrency}")
    print("=" * 70)

    all_results = {}
    for method in args.methods:
        print(f"\nEvaluating method: {method}...")
        results = await evaluate_method(
            client,
            args.model,
            dataset,
            method,
            gepa_prompt,
            args.num_examples,
            args.max_tokens,
            args.temperature,
            args.concurrency,
        )

        scores = [r["score"] for r in results]
        correct = sum(1 for s in scores if s > 0)
        total = len(scores)
        accuracy = correct / total if total > 0 else 0

        errors = sum(1 for r in results if "error" in r)
        no_answer = sum(1 for r in results if not r.get("has_answer_tags", True))

        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"  Errors: {errors}, No answer tags: {no_answer}")

        all_results[method] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "errors": errors,
            "no_answer_tags": no_answer,
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for method, res in all_results.items():
        print(f"  {method:25s}: {res['accuracy']:.4f} ({res['correct']}/{res['total']})")

    # Save results
    output_path = Path("eval_standalone_results.json")
    with open(output_path, "w") as f:
        json.dump({"config": vars(args), "results": all_results}, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
