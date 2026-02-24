#!/usr/bin/env python3
"""Generate TOML overlay files for each GEPA loop iteration.

Each overlay sets iteration-specific fields (max_steps, resume_step, teacher
context) that override the base rl_gepa_loop.toml config.

Usage:
    python generate_gepa_overlay.py \
      --iteration 1 \
      --steps-per-iter 150 \
      --prompt-file /path/to/teacher_prompt.txt \
      --output overlay_iter1.toml \
      --base-config experiments/context_distill/rl_gepa_loop_s42.toml
"""

import argparse
from pathlib import Path

import tomli
import tomli_w


def generate_overlay(
    iteration: int,
    steps_per_iter: int,
    prompt_file: str,
    output: str,
    base_config: str,
) -> None:
    max_steps = (iteration + 1) * steps_per_iter
    resume_step = iteration * steps_per_iter

    # Read teacher prompt
    prompt_text = Path(prompt_file).read_text().strip()

    # Read base config to extract orchestrator settings that must be preserved
    with open(base_config, "rb") as f:
        base = tomli.load(f)
    base_orch = base.get("orchestrator", {})

    # Build overlay dict
    # NOTE: pydantic-settings does shallow merge on TOML sections. Any top-level
    # section present in the overlay REPLACES the entire section from the base config.
    # To avoid losing settings, we start from the base config's values and only
    # override what changes per iteration (teacher_model.context, max_steps, resume_step).
    overlay_orch = dict(base_orch)
    overlay_orch["teacher_model"] = {"context": prompt_text}

    overlay: dict = {
        "max_steps": max_steps,
        "orchestrator": overlay_orch,
    }

    # Preserve ckpt section from base, adding resume_step for iterations > 0
    if iteration > 0:
        base_ckpt = dict(base.get("ckpt", {}))
        base_ckpt["resume_step"] = resume_step
        overlay["ckpt"] = base_ckpt

    # Write TOML
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        tomli_w.dump(overlay, f)

    print(f"Generated overlay: {out_path}")
    print(f"  iteration={iteration}, max_steps={max_steps}, resume_step={resume_step if iteration > 0 else 'N/A'}")
    print(f"  prompt_file={prompt_file} ({len(prompt_text)} chars)")


def main():
    parser = argparse.ArgumentParser(description="Generate GEPA loop TOML overlay")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration number (0-indexed)")
    parser.add_argument("--steps-per-iter", type=int, required=True, help="RL steps per iteration")
    parser.add_argument("--prompt-file", type=str, required=True, help="Path to teacher prompt text file")
    parser.add_argument("--output", type=str, required=True, help="Output TOML file path")
    parser.add_argument("--base-config", type=str, required=True, help="Base TOML config to read orchestrator env/eval/log from")
    args = parser.parse_args()

    generate_overlay(
        iteration=args.iteration,
        steps_per_iter=args.steps_per_iter,
        prompt_file=args.prompt_file,
        output=args.output,
        base_config=args.base_config,
    )


if __name__ == "__main__":
    main()
