# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)

from sglang_omni.diffusion.flow_unipc import FlowUniPCSolver


def test_flow_unipc_solver_matches_sglang_scheduler_step_for_step() -> None:
    num_steps = 3
    shift = 5.0
    solver = FlowUniPCSolver.create(
        num_inference_steps=num_steps,
        device="cpu",
        shift=shift,
    )
    reference = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=shift,
    )
    reference.set_timesteps(num_steps, device="cpu", shift=shift)

    actual = torch.linspace(-1.0, 1.0, 16).reshape(1, 2, 2, 2, 2)
    expected = actual.clone()
    assert torch.equal(solver.timesteps, reference.timesteps)
    assert torch.equal(solver.sigmas, reference.sigmas)

    for timestep in reference.timesteps:
        actual_velocity = actual * 0.125 + 0.01
        expected_velocity = expected * 0.125 + 0.01
        assert torch.equal(solver.timestep, timestep)
        actual = solver.step(actual_velocity, actual)
        expected = reference.step(
            model_output=expected_velocity,
            timestep=timestep,
            sample=expected,
            return_dict=False,
        )[0]
        torch.testing.assert_close(actual, expected)

    assert solver.step_index == num_steps
    assert solver.is_finished
    with pytest.raises(RuntimeError, match="already finished"):
        solver.step(torch.zeros_like(actual), actual)


def test_flow_unipc_solver_instances_do_not_share_history() -> None:
    first = FlowUniPCSolver.create(num_inference_steps=2, device="cpu", shift=1.0)
    second = FlowUniPCSolver.create(num_inference_steps=2, device="cpu", shift=1.0)
    sample = torch.zeros(1, 1, 2, 2)

    first.step(torch.ones_like(sample), sample)

    assert first.step_index == 1
    assert second.step_index == 0


def test_flow_unipc_solver_rejects_empty_schedule() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        FlowUniPCSolver.create(num_inference_steps=0, device="cpu")
