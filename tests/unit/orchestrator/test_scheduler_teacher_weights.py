"""Tests that the Scheduler updates teacher weights alongside inference weights when share_teacher_weights is enabled."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from prime_rl.orchestrator.config import SamplingConfig
from prime_rl.orchestrator.scheduler import Scheduler


def make_scheduler(teacher_admin_clients=None, teacher_lora_name=None):
    """Create a Scheduler with minimal mocked dependencies."""
    config = MagicMock()
    config.output_dir = Path("/tmp/test_output")
    config.batch_size = 8
    config.rollouts_per_example = 2
    config.seq_len = 1024
    config.max_steps = 10
    config.sampling = SamplingConfig(temperature=1.0)
    config.trajectory_strategy = "interleaved"
    config.log.env_worker_logs = False
    config.workers_per_env = 1
    config.max_concurrent = None

    inference_pool = AsyncMock()
    inference_pool.update_weights = AsyncMock()
    inference_pool.get_metrics.return_value = {}

    buffer = MagicMock()

    scheduler = Scheduler(
        client_config=MagicMock(),
        env_configs=[],
        buffer=buffer,
        config=config,
        oversampling_factor=1.0,
        max_async_level=2,
        max_off_policy_steps=2,
        strict_async_level=False,
        inference_pool=inference_pool,
        lora_name=None,
        output_dir=Path("/tmp/test_output"),
        teacher_admin_clients=teacher_admin_clients,
        teacher_lora_name=teacher_lora_name,
    )
    return scheduler


class TestTeacherWeightUpdates:
    """Tests for teacher weight sharing in Scheduler.update_policy()."""

    def test_no_teacher_clients_only_updates_inference(self):
        """When teacher_admin_clients is None, only inference pool gets updated."""
        scheduler = make_scheduler(teacher_admin_clients=None)
        scheduler.step = 1
        scheduler.ckpt_step = 0

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=1),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new_callable=AsyncMock),
            patch("prime_rl.orchestrator.scheduler.update_weights", new_callable=AsyncMock) as mock_update_teacher,
        ):
            asyncio.run(scheduler.update_policy())

        scheduler.inference_pool.update_weights.assert_called_once()
        mock_update_teacher.assert_not_called()

    def test_teacher_clients_updates_both(self):
        """When teacher_admin_clients is set, both inference and teacher get updated."""
        teacher_clients = [MagicMock()]
        scheduler = make_scheduler(teacher_admin_clients=teacher_clients)
        scheduler.step = 1
        scheduler.ckpt_step = 0

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=1),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new_callable=AsyncMock),
            patch("prime_rl.orchestrator.scheduler.update_weights", new_callable=AsyncMock) as mock_update_teacher,
        ):
            asyncio.run(scheduler.update_policy())

        # Both should be called
        scheduler.inference_pool.update_weights.assert_called_once()
        mock_update_teacher.assert_called_once()

        # Teacher should get same weights path and step
        call_args = mock_update_teacher.call_args
        assert call_args[0][0] == teacher_clients  # admin_clients
        assert call_args[1]["step"] == 1  # same step

    def test_teacher_lora_name_passed(self):
        """When teacher_lora_name is set, it's passed to update_weights."""
        teacher_clients = [MagicMock()]
        scheduler = make_scheduler(teacher_admin_clients=teacher_clients, teacher_lora_name="r16-a64")
        scheduler.step = 1
        scheduler.ckpt_step = 0

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=1),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new_callable=AsyncMock),
            patch("prime_rl.orchestrator.scheduler.update_weights", new_callable=AsyncMock) as mock_update_teacher,
        ):
            asyncio.run(scheduler.update_policy())

        call_args = mock_update_teacher.call_args
        assert call_args[1]["lora_name"] == "r16-a64"

    def test_no_update_when_no_new_checkpoint(self):
        """When no new checkpoint is available, neither should be updated."""
        teacher_clients = [MagicMock()]
        scheduler = make_scheduler(teacher_admin_clients=teacher_clients)
        scheduler.step = 0
        scheduler.ckpt_step = 0

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=0),
            patch("prime_rl.orchestrator.scheduler.update_weights", new_callable=AsyncMock) as mock_update_teacher,
        ):
            asyncio.run(scheduler.update_policy())

        scheduler.inference_pool.update_weights.assert_not_called()
        mock_update_teacher.assert_not_called()
