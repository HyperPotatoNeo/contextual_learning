import torch

from prime_rl.orchestrator.config import AdvantageConfig


def compute_advantages(
    rewards: list[float],
    completion_lengths: list[int],
    samples_per_problem: int,
    advantage_config: AdvantageConfig | None,
    teacher_logprobs: list[list[float]] | None = None,
    inference_logprobs: list[list[float]] | None = None,
) -> list[float]:
    """
    Computes advantages from a flattened list of rewards, grouped by problem.

    Args:
        rewards: Flattened list of rewards where first `samples_per_problem` rewards are for the first problem
        completion_lengths: List of completion lengths for each reward
        samples_per_problem: Number of samples (and thus, rewards) per problem
        advantage_config: Configuration for advantage computation
        teacher_logprobs: Optional list of per-token teacher logprobs for each sample (for full reward baseline)
        inference_logprobs: Optional list of per-token inference logprobs for each sample (for full reward baseline)
    """
    if not advantage_config:
        return rewards

    rewards_tensor = torch.tensor(rewards).view(-1, samples_per_problem)
    lengths = torch.tensor(completion_lengths).view(-1, samples_per_problem)

    # Compute full reward signal if configured
    if advantage_config.use_full_reward_baseline and teacher_logprobs is not None and inference_logprobs is not None:
        # Sum logprobs over completion tokens to get per-sample scalars
        teacher_sums = torch.tensor([sum(lp) for lp in teacher_logprobs]).view(-1, samples_per_problem)
        inference_sums = torch.tensor([sum(lp) for lp in inference_logprobs]).view(-1, samples_per_problem)

        # Full reward: adv_tau * task_reward + teacher_tau * sum(teacher_lp) - student_tau * sum(inference_lp)
        full_rewards = (
            advantage_config.adv_tau * rewards_tensor
            + advantage_config.teacher_tau * teacher_sums
            - advantage_config.student_tau * inference_sums
        )
    else:
        full_rewards = rewards_tensor

    # Compute baseline
    if advantage_config.length_weighted_mean:
        baseline = (full_rewards * lengths).sum(dim=1, keepdim=True) / lengths.sum(dim=1, keepdim=True)
    else:
        baseline = full_rewards.mean(dim=1, keepdim=True)

    return (full_rewards - baseline).flatten().tolist()
