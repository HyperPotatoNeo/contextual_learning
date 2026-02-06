import pytest
import torch

from prime_rl.trainer.rl.config import LossConfig
from prime_rl.trainer.rl.loss import compute_entropy, compute_loss

pytestmark = [pytest.mark.gpu]


def test_grpo_loss():
    trainer_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    teacher_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    advantages = [torch.randn(50).cuda(), torch.randn(30).cuda()]
    loss_mask = [torch.ones(50, dtype=torch.bool).cuda(), torch.ones(30, dtype=torch.bool).cuda()]

    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        teacher_logprobs,
        advantages,
        loss_mask=loss_mask,
        loss_config=LossConfig(ratio_type="token", token_mask_high=10.0),
        loss_scale=1.0,
    )
    assert loss.shape == ()


def test_gspo_loss():
    # Create list of tensors as expected by compute_loss (simulating split sequences)
    trainer_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    teacher_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    advantages = [torch.randn(40).cuda(), torch.randn(60).cuda()]
    loss_mask = [torch.ones(40, dtype=torch.bool).cuda(), torch.ones(60, dtype=torch.bool).cuda()]

    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        teacher_logprobs,
        advantages,
        loss_mask=loss_mask,
        loss_config=LossConfig(ratio_type="sequence", token_mask_high=10.0),
        loss_scale=1.0,
    )
    assert loss.shape == ()


def test_entropy_loss():
    shifted_logits = torch.randn(10, 10, 10, dtype=torch.float32).cuda()
    entropy = compute_entropy(shifted_logits)
    assert entropy.shape == (10, 10)


class TestPerTokenKLBehavior:
    """Tests for per-token KL behavior when use_full_reward_baseline is False."""

    def test_per_token_kl_added_when_use_full_reward_baseline_false(self):
        """When use_full_reward_baseline=false and teacher_logprobs provided, per-token KL should be added."""
        seq_len = 10
        trainer_logprobs = [torch.zeros(seq_len, dtype=torch.float32).cuda()]
        inference_logprobs = [torch.zeros(seq_len, dtype=torch.float32).cuda()]
        teacher_logprobs = [torch.ones(seq_len, dtype=torch.float32).cuda()]  # Teacher logprobs = 1.0
        advantages = [torch.zeros(seq_len).cuda()]  # Base advantage = 0
        loss_mask = [torch.ones(seq_len, dtype=torch.bool).cuda()]

        # When use_full_reward_baseline=False, loss.py should add per-token KL:
        # adv_tau * advantage + teacher_tau * teacher_lp - student_tau * trainer_lp
        loss_config = LossConfig(
            use_full_reward_baseline=False,
            adv_tau=1.0,
            teacher_tau=0.5,  # Will add 0.5 * 1.0 = 0.5 per token
            student_tau=0.0,  # No entropy term
            token_mask_high=100.0,
            token_mask_low=0.01,
        )

        loss, metrics = compute_loss(
            trainer_logprobs,
            inference_logprobs,
            teacher_logprobs,
            advantages,
            loss_mask=loss_mask,
            loss_config=loss_config,
            loss_scale=1.0,
        )

        # The combined_advantage should reflect the added KL term
        # Base advantage = 0, teacher_tau * teacher_lp = 0.5 * 1.0 = 0.5
        # So combined_advantage should be close to 0.5
        assert metrics["combined_advantage"].mean().item() == pytest.approx(0.5, abs=0.01)

    def test_no_kl_when_use_full_reward_baseline_true(self):
        """When use_full_reward_baseline=true, advantage should be used directly without modification."""
        seq_len = 10
        trainer_logprobs = [torch.zeros(seq_len, dtype=torch.float32).cuda()]
        inference_logprobs = [torch.zeros(seq_len, dtype=torch.float32).cuda()]
        teacher_logprobs = [torch.ones(seq_len, dtype=torch.float32).cuda()]
        base_advantage = 2.0
        advantages = [torch.full((seq_len,), base_advantage).cuda()]
        loss_mask = [torch.ones(seq_len, dtype=torch.bool).cuda()]

        # When use_full_reward_baseline=True, advantage is used as-is (no scaling or KL addition)
        loss_config = LossConfig(
            use_full_reward_baseline=True,
            adv_tau=0.5,  # Should NOT be applied
            teacher_tau=1.0,  # Should NOT be applied
            student_tau=1.0,  # Should NOT be applied
            token_mask_high=100.0,
            token_mask_low=0.01,
        )

        loss, metrics = compute_loss(
            trainer_logprobs,
            inference_logprobs,
            teacher_logprobs,
            advantages,
            loss_mask=loss_mask,
            loss_config=loss_config,
            loss_scale=1.0,
        )

        # Combined advantage should equal the base advantage (no scaling/KL)
        assert metrics["combined_advantage"].mean().item() == pytest.approx(base_advantage, abs=0.01)

    def test_no_kl_when_teacher_logprobs_none(self):
        """Normal RL without teacher model: no KL terms should be added."""
        seq_len = 10
        trainer_logprobs = [torch.zeros(seq_len, dtype=torch.float32).cuda()]
        inference_logprobs = [torch.zeros(seq_len, dtype=torch.float32).cuda()]
        teacher_logprobs = None  # No teacher model
        base_advantage = 1.0
        advantages = [torch.full((seq_len,), base_advantage).cuda()]
        loss_mask = [torch.ones(seq_len, dtype=torch.bool).cuda()]

        loss_config = LossConfig(
            use_full_reward_baseline=False,
            adv_tau=2.0,  # Should be applied
            teacher_tau=1.0,  # Should NOT be applied (no teacher)
            student_tau=1.0,  # Should NOT be applied (no teacher)
            token_mask_high=100.0,
            token_mask_low=0.01,
        )

        loss, metrics = compute_loss(
            trainer_logprobs,
            inference_logprobs,
            teacher_logprobs,
            advantages,
            loss_mask=loss_mask,
            loss_config=loss_config,
            loss_scale=1.0,
        )

        # Combined advantage should be adv_tau * base_advantage = 2.0 * 1.0 = 2.0
        assert metrics["combined_advantage"].mean().item() == pytest.approx(2.0, abs=0.01)
