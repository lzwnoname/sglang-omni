# SPDX-License-Identifier: Apache-2.0
"""Request-local adapter for SGLang's flow-matching UniPC solver."""

from __future__ import annotations

from typing import Any

import torch


def _load_scheduler_class() -> type[Any]:
    # Keep the import lazy: importing SGLang's diffusion package initializes
    # optional accelerator backends that ordinary Omni stages do not need.
    from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
        FlowUniPCMultistepScheduler,
    )

    return FlowUniPCMultistepScheduler


class FlowUniPCSolver:
    """Own one mutable FlowUniPC chain for one request and one modality.

    FlowUniPC keeps multistep history on the scheduler instance. Callers must
    therefore create separate ``FlowUniPCSolver`` objects for concurrent
    requests and for independently shaped modalities of the same request.
    """

    def __init__(self, scheduler: Any) -> None:
        self._scheduler = scheduler

    @classmethod
    def create(
        cls,
        *,
        num_inference_steps: int,
        device: str | torch.device,
        shift: float = 1.0,
        num_train_timesteps: int = 1000,
        **scheduler_kwargs: Any,
    ) -> FlowUniPCSolver:
        """Create and initialize a request-local SGLang FlowUniPC solver."""

        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")

        scheduler_class = _load_scheduler_class()
        scheduler = scheduler_class(
            num_train_timesteps=num_train_timesteps,
            shift=float(shift),
            **scheduler_kwargs,
        )
        scheduler.set_timesteps(
            num_inference_steps,
            device=device,
            shift=float(shift),
        )
        return cls(scheduler)

    @property
    def timesteps(self) -> torch.Tensor:
        return self._scheduler.timesteps

    @property
    def sigmas(self) -> torch.Tensor:
        return self._scheduler.sigmas

    @property
    def step_index(self) -> int:
        index = self._scheduler.step_index
        return 0 if index is None else int(index)

    @property
    def is_finished(self) -> bool:
        return self.step_index >= len(self.timesteps)

    @property
    def timestep(self) -> torch.Tensor:
        if self.is_finished:
            raise RuntimeError("FlowUniPC solver has already finished")
        return self.timesteps[self.step_index]

    def step(
        self,
        velocity: torch.Tensor,
        sample: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Advance one FlowUniPC step using a flow-prediction velocity."""

        timestep = self.timestep
        return self._scheduler.step(
            model_output=velocity,
            timestep=timestep,
            sample=sample,
            return_dict=False,
            generator=generator,
        )[0]

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Delegate scheduler-consistent noising for conditioned inputs."""

        return self._scheduler.add_noise(original_samples, noise, timesteps)
