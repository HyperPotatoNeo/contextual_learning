import pytest

from prime_rl.orchestrator.advantage import compute_advantages
from prime_rl.orchestrator.config import AdvantageConfig


class TestComputeAdvantages:
    """Tests for the compute_advantages function."""

    def test_no_config_returns_rewards(self):
        """When advantage_config is None, should return raw rewards."""
        rewards = [1.0, 2.0, 3.0, 4.0]
        completion_lengths = [10, 20, 30, 40]
        result = compute_advantages(rewards, completion_lengths, 2, None)
        assert result == rewards

    def test_basic_baseline_subtraction(self):
        """Test basic mean baseline subtraction."""
        # 2 problems, 2 samples each
        rewards = [1.0, 3.0, 2.0, 4.0]  # Problem 1: [1, 3], Problem 2: [2, 4]
        completion_lengths = [10, 10, 10, 10]
        config = AdvantageConfig()

        result = compute_advantages(rewards, completion_lengths, 2, config)

        # Problem 1: mean=2, advantages=[-1, 1]
        # Problem 2: mean=3, advantages=[-1, 1]
        expected = [-1.0, 1.0, -1.0, 1.0]
        assert result == pytest.approx(expected)

    def test_length_weighted_mean(self):
        """Test length-weighted mean baseline."""
        # Problem with samples of different lengths
        rewards = [1.0, 2.0]  # 1 problem, 2 samples
        completion_lengths = [10, 30]  # Second sample has 3x the length
        config = AdvantageConfig(length_weighted_mean=True)

        result = compute_advantages(rewards, completion_lengths, 2, config)

        # Length-weighted mean: (1*10 + 2*30) / (10 + 30) = 70/40 = 1.75
        # Advantages: [1 - 1.75, 2 - 1.75] = [-0.75, 0.25]
        expected = [-0.75, 0.25]
        assert result == pytest.approx(expected)

    def test_full_reward_baseline_without_logprobs(self):
        """When use_full_reward_baseline is True but no logprobs provided, should use task rewards only."""
        rewards = [1.0, 3.0, 2.0, 4.0]
        completion_lengths = [10, 10, 10, 10]
        config = AdvantageConfig(
            use_full_reward_baseline=True,
            adv_tau=1.0,
            teacher_tau=0.5,
            student_tau=0.5,
        )

        result = compute_advantages(rewards, completion_lengths, 2, config)

        # Without teacher/inference logprobs, should fall back to task rewards
        expected = [-1.0, 1.0, -1.0, 1.0]
        assert result == pytest.approx(expected)

    def test_full_reward_baseline_with_logprobs(self):
        """Test full reward baseline with teacher and inference logprobs."""
        # 1 problem, 2 samples
        rewards = [1.0, 1.0]  # Same task reward
        completion_lengths = [3, 3]

        # Teacher logprobs: sample 1 has higher teacher prob
        teacher_logprobs = [
            [-1.0, -1.0, -1.0],  # sum = -3
            [-2.0, -2.0, -2.0],  # sum = -6
        ]
        # Inference logprobs: same for both
        inference_logprobs = [
            [-1.5, -1.5, -1.5],  # sum = -4.5
            [-1.5, -1.5, -1.5],  # sum = -4.5
        ]

        config = AdvantageConfig(
            use_full_reward_baseline=True,
            adv_tau=1.0,
            teacher_tau=1.0,
            student_tau=1.0,
        )

        result = compute_advantages(
            rewards,
            completion_lengths,
            2,
            config,
            teacher_logprobs=teacher_logprobs,
            inference_logprobs=inference_logprobs,
        )

        # Full reward = adv_tau * reward + teacher_tau * sum(teacher_lp) - student_tau * sum(inference_lp)
        # Sample 1: 1.0 * 1.0 + 1.0 * (-3) - 1.0 * (-4.5) = 1 - 3 + 4.5 = 2.5
        # Sample 2: 1.0 * 1.0 + 1.0 * (-6) - 1.0 * (-4.5) = 1 - 6 + 4.5 = -0.5
        # Baseline: (2.5 + (-0.5)) / 2 = 1.0
        # Advantages: [2.5 - 1.0, -0.5 - 1.0] = [1.5, -1.5]
        expected = [1.5, -1.5]
        assert result == pytest.approx(expected)

    def test_full_reward_baseline_with_tau_scaling(self):
        """Test that tau values correctly scale the components."""
        rewards = [2.0, 0.0]  # 1 problem, 2 samples
        completion_lengths = [2, 2]

        teacher_logprobs = [
            [-1.0, -1.0],  # sum = -2
            [-1.0, -1.0],  # sum = -2
        ]
        inference_logprobs = [
            [-2.0, -2.0],  # sum = -4
            [-2.0, -2.0],  # sum = -4
        ]

        # With adv_tau=0.5, teacher and inference logprobs should cancel out
        # and we should get 0.5 * task_advantage
        config = AdvantageConfig(
            use_full_reward_baseline=True,
            adv_tau=0.5,
            teacher_tau=1.0,
            student_tau=1.0,
        )

        result = compute_advantages(
            rewards,
            completion_lengths,
            2,
            config,
            teacher_logprobs=teacher_logprobs,
            inference_logprobs=inference_logprobs,
        )

        # Full reward = 0.5 * reward + 1.0 * (-2) - 1.0 * (-4) = 0.5 * reward + 2
        # Sample 1: 0.5 * 2 + 2 = 3
        # Sample 2: 0.5 * 0 + 2 = 2
        # Baseline: (3 + 2) / 2 = 2.5
        # Advantages: [3 - 2.5, 2 - 2.5] = [0.5, -0.5]
        expected = [0.5, -0.5]
        assert result == pytest.approx(expected)

    def test_advantages_sum_to_zero_within_group(self):
        """Verify that advantages sum to zero within each problem group."""
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # 2 problems, 3 samples each
        completion_lengths = [10, 10, 10, 10, 10, 10]
        config = AdvantageConfig()

        result = compute_advantages(rewards, completion_lengths, 3, config)

        # Check that each group sums to ~0
        group1_sum = sum(result[0:3])
        group2_sum = sum(result[3:6])
        assert group1_sum == pytest.approx(0.0)
        assert group2_sum == pytest.approx(0.0)

    def test_full_reward_baseline_advantages_sum_to_zero(self):
        """Full reward baseline advantages should also sum to zero within groups."""
        rewards = [1.0, 2.0, 3.0, 4.0]  # 2 problems, 2 samples each
        completion_lengths = [5, 5, 5, 5]

        teacher_logprobs = [
            [-1.0] * 5,
            [-2.0] * 5,
            [-1.5] * 5,
            [-2.5] * 5,
        ]
        inference_logprobs = [
            [-0.5] * 5,
            [-1.0] * 5,
            [-0.7] * 5,
            [-1.2] * 5,
        ]

        config = AdvantageConfig(
            use_full_reward_baseline=True,
            adv_tau=0.5,
            teacher_tau=0.5,
            student_tau=0.5,
        )

        result = compute_advantages(
            rewards,
            completion_lengths,
            2,
            config,
            teacher_logprobs=teacher_logprobs,
            inference_logprobs=inference_logprobs,
        )

        # Check that each group sums to ~0
        group1_sum = sum(result[0:2])
        group2_sum = sum(result[2:4])
        assert group1_sum == pytest.approx(0.0)
        assert group2_sum == pytest.approx(0.0)


class TestRLConfigTauSync:
    """Tests for tau value syncing from trainer.loss to orchestrator.advantage."""

    def test_tau_values_synced_from_trainer_loss(self):
        """When use_full_reward_baseline is True, tau values should be copied from trainer.loss."""
        from prime_rl.rl import RLConfig

        config = RLConfig(
            trainer={
                "loss": {
                    "adv_tau": 0.3,
                    "teacher_tau": 0.4,
                    "student_tau": 0.5,
                }
            },
            orchestrator={
                "advantage": {
                    "use_full_reward_baseline": True,
                },
                "teacher_model": {},  # Required when teacher_tau > 0
            },
            teacher_gpu_ids=[1],  # Required to enable teacher model
        )

        # Tau values should be synced from trainer.loss
        assert config.orchestrator.advantage.adv_tau == 0.3
        assert config.orchestrator.advantage.teacher_tau == 0.4
        assert config.orchestrator.advantage.student_tau == 0.5

    def test_tau_values_not_synced_when_full_reward_baseline_disabled(self):
        """When use_full_reward_baseline is False, tau values should remain at defaults."""
        from prime_rl.rl import RLConfig

        config = RLConfig(
            trainer={
                "loss": {
                    "adv_tau": 0.3,
                    "teacher_tau": 0.4,
                    "student_tau": 0.5,
                }
            },
            orchestrator={
                "advantage": {
                    "use_full_reward_baseline": False,
                },
                "teacher_model": {},  # Required when teacher_tau > 0
            },
            teacher_gpu_ids=[1],  # Required to enable teacher model
        )

        # Tau values should remain at defaults (not synced)
        assert config.orchestrator.advantage.adv_tau == 1.0
        assert config.orchestrator.advantage.teacher_tau == 0.0
        assert config.orchestrator.advantage.student_tau == 0.0
