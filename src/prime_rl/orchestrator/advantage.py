import torch

from prime_rl.orchestrator.config import AdvantageConfig


def compute_kl_gates(rewards: list[float], samples_per_problem: int) -> list[float]:
    """Compute per-sample KL gates based on group task reward mean.

    Returns (1 - group_mean) for each sample, where group_mean is the mean task reward
    of all samples in the same problem group. This gates KL terms so they are only active
    when the group lacks task reward signal.
    """
    rewards_tensor = torch.tensor(rewards).view(-1, samples_per_problem)
    group_mean = rewards_tensor.mean(dim=1, keepdim=True)
    kl_gate = (1 - group_mean).expand_as(rewards_tensor)
    return kl_gate.flatten().tolist()


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

        kl_terms = advantage_config.teacher_tau * teacher_sums - advantage_config.student_tau * inference_sums

        # Gate KL terms by (1 - group_task_mean) when kl_only_incorrect is enabled
        if advantage_config.kl_only_incorrect:
            group_task_mean = rewards_tensor.mean(dim=1, keepdim=True)
            kl_terms = (1 - group_task_mean) * kl_terms

        # Full reward: adv_tau * task_reward + (gated) KL terms
        full_rewards = advantage_config.adv_tau * rewards_tensor + kl_terms
    else:
        full_rewards = rewards_tensor

    # Compute baseline
    if advantage_config.length_weighted_mean:
        baseline = (full_rewards * lengths).sum(dim=1, keepdim=True) / lengths.sum(dim=1, keepdim=True)
    else:
        baseline = full_rewards.mean(dim=1, keepdim=True)

    return (full_rewards - baseline).flatten().tolist()
