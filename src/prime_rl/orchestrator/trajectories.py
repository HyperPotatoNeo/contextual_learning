from copy import deepcopy
from typing import Any

import verifiers as vf

from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger


def _extract_prompt_text(prompt: Any) -> str | None:
    """
    Extract prompt text from either string or chat message format.

    Args:
        prompt: Either a string or a list of chat messages like [{"role": "user", "content": "..."}]

    Returns:
        The prompt as a string, or None if extraction fails.
    """
    if prompt is None:
        return None
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        # Chat format: extract user message content
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content")
        # Fallback: concatenate all content
        contents = [msg.get("content", "") for msg in prompt if isinstance(msg, dict)]
        return "\n".join(contents) if contents else None
    return None


def interleave_rollout(state: vf.State) -> list[TrainingSample] | None:
    """
    Convert vf.State to a *single* trainable rollout by interleaving the trajectory.

    NOTE:
    - This requires that consecutive trajectory steps share token prefixes (incremental tokenization)
    - This approach is suceptible to introduce subtle difference due to re-tokenization in multi-turn environments.
    """
    logger = get_logger()

    trajectory = state["trajectory"]
    if len(trajectory) == 0:
        logger.warning(f"No trajectory steps for example {state['example_id']}. Skipping rollout.")
        return None

    has_error = state["error"] is not None

    # Initialize the rollout with prompt and completion from first trajectory step
    first_step = trajectory[0]
    temperature = first_step["temperature"]
    if has_error:
        completion_mask = [False] * len(first_step["tokens"]["completion_mask"])
    else:
        completion_mask = [bool(i) for i in first_step["tokens"]["completion_mask"]]
    completion_ids = deepcopy(first_step["tokens"]["completion_ids"])
    interleaved_rollout = TrainingSample(
        prompt_ids=deepcopy(first_step["tokens"]["prompt_ids"]),
        prompt_mask=[bool(i) for i in first_step["tokens"]["prompt_mask"]],
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        completion_logprobs=deepcopy(first_step["tokens"]["completion_logprobs"]),
        completion_temperatures=[temperature] * len(completion_ids),  # Per-token temperatures
        teacher_logprobs=None,  # Populated at the end after full sequence length is known if teacher model is configured
        advantage=None,
        prompt_text=_extract_prompt_text(first_step.get("prompt")),  # For context distillation
    )

    # Interleave all other trajectory steps into completion
    prefix_tokens = first_step["tokens"]["prompt_ids"] + first_step["tokens"]["completion_ids"]
    for step_idx, step in enumerate(trajectory[1:], start=2):
        tokens = step["tokens"]
        step_temperature = step["temperature"]
        assert tokens is not None
        prev_trajectory_and_new_prompt_ids = tokens["prompt_ids"]

        # Incremental tokenization assumption
        if not prefix_tokens == prev_trajectory_and_new_prompt_ids[: len(prefix_tokens)]:
            logger.warning(
                f"Found mismatch in prefix tokens for example {state['example_id']} at trajectory step {step_idx}"
            )

        # Extend the completion with the new prompt (use step's temperature for prompt tokens too)
        prompt_ids = deepcopy(prev_trajectory_and_new_prompt_ids[len(prefix_tokens) :])
        interleaved_rollout.completion_ids.extend(prompt_ids)
        interleaved_rollout.completion_mask.extend([False] * len(prompt_ids))
        interleaved_rollout.completion_logprobs.extend([0.0] * len(prompt_ids))
        interleaved_rollout.completion_temperatures.extend([step_temperature] * len(prompt_ids))

        # Extend the completion with the new completion tokens
        completion_ids = deepcopy(tokens["completion_ids"])
        completion_logprobs = deepcopy(tokens["completion_logprobs"])
        interleaved_rollout.completion_ids.extend(completion_ids)
        if has_error:
            interleaved_rollout.completion_mask.extend([False] * len(tokens["completion_mask"]))
        else:
            interleaved_rollout.completion_mask.extend([bool(i) for i in tokens["completion_mask"]])
        interleaved_rollout.completion_logprobs.extend(completion_logprobs)
        interleaved_rollout.completion_temperatures.extend([step_temperature] * len(completion_ids))

        # New prefix is the current prompt and completion ids concatenated
        prefix_tokens = tokens["prompt_ids"] + tokens["completion_ids"]

    return [interleaved_rollout]


def branch_rollout(state: vf.State) -> list[TrainingSample] | None:
    """Convert vf.State to *multiple* trainable rollouts using branching trajectories strategy."""
    logger = get_logger()

    rollouts = []
    trajectory = state["trajectory"]
    if len(trajectory) == 0:
        logger.warning(f"No trajectory steps for example {state['example_id']}. Skipping rollout.")
        return None

    has_error = state["error"] is not None
    for step in state["trajectory"]:
        assert "tokens" in step
        tokens = step["tokens"]
        temperature = step["temperature"]
        if has_error:
            completion_mask = [False] * len(tokens["completion_mask"])
        else:
            completion_mask = [bool(i) for i in tokens["completion_mask"]]
        completion_ids = deepcopy(tokens["completion_ids"])
        rollout = TrainingSample(
            prompt_ids=deepcopy(tokens["prompt_ids"]),
            prompt_mask=[bool(i) for i in tokens["prompt_mask"]],
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            completion_logprobs=deepcopy(tokens["completion_logprobs"]),
            completion_temperatures=[temperature] * len(completion_ids),  # Per-token temperatures
            advantage=step.get("advantage"),  # Pre-computed by env (e.g. RSAgent per_step_grpo)
            reward=step.get("reward"),  # Pre-computed by env (e.g. RSAgent per_step_grpo)
            teacher_logprobs=None,
            prompt_text=_extract_prompt_text(step.get("prompt")),  # For context distillation
        )
        rollouts.append(rollout)
    return rollouts
