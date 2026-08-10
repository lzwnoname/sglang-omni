# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
import threading
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from sglang_omni.scheduling import continuous_diffusion_scheduler as scheduler_module
from sglang_omni.scheduling.continuous_diffusion_scheduler import (
    ContinuousDiffusionScheduler,
)
from sglang_omni.scheduling.messages import IncomingMessage
from tests.unit_test.pipeline.helpers import run_scheduler


@dataclass
class _State:
    name: str
    batch_key: str
    total_steps: int
    steps: int = 0


class _Backend:
    def __init__(self) -> None:
        self.step_batches: list[tuple[str, ...]] = []
        self.released: list[str] = []

    def prepare(self, payload: dict[str, Any]) -> _State:
        if payload.get("prepare_error"):
            raise RuntimeError(f"prepare {payload['name']}")
        return _State(
            name=payload["name"],
            batch_key=payload.get("batch_key", "default"),
            total_steps=payload.get("total_steps", 1),
        )

    def batch_key(self, state: _State) -> str:
        return state.batch_key

    def step(self, states: list[_State]) -> None:
        self.step_batches.append(tuple(state.name for state in states))
        for state in states:
            state.steps += 1

    def is_finished(self, state: _State) -> bool:
        return state.steps >= state.total_steps

    def finalize(self, state: _State) -> dict[str, Any]:
        return {"name": state.name, "steps": state.steps}

    def release(self, state: _State) -> None:
        self.released.append(state.name)


def _tp_group(rank: int = 0, world_size: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        rank=rank,
        rank_in_group=rank,
        world_size=world_size,
        ranks=list(range(world_size)),
        cpu_group=object(),
    )


def _scheduler(
    backend: _Backend,
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
    **kwargs: Any,
) -> ContinuousDiffusionScheduler:
    return ContinuousDiffusionScheduler(
        backend,
        tp_group=_tp_group(rank=tp_rank, world_size=tp_size),
        **kwargs,
    )


def _message(request_id: str, **payload: Any) -> IncomingMessage:
    return IncomingMessage(
        request_id=request_id,
        type="new_request",
        data={"name": request_id, **payload},
    )


def test_scheduler_batches_compatible_states_and_round_robins_batch_keys() -> None:
    backend = _Backend()
    scheduler = _scheduler(
        backend,
        max_batch_size=2,
        max_running_requests=3,
        max_batch_wait_ms=20,
    )

    outputs = run_scheduler(
        scheduler,
        [
            _message("image-a", batch_key="image", total_steps=2),
            _message("image-b", batch_key="image", total_steps=2),
            _message("video", batch_key="video", total_steps=1),
        ],
        output_count=3,
    )

    assert backend.step_batches == [
        ("image-a", "image-b"),
        ("video",),
        ("image-a", "image-b"),
    ]
    assert {output.request_id for output in outputs} == {
        "image-a",
        "image-b",
        "video",
    }
    assert all(output.type == "result" for output in outputs)
    assert sorted(backend.released) == ["image-a", "image-b", "video"]


def test_tp_leader_owns_ingress_batch_wait_and_output(monkeypatch) -> None:
    broadcast_payload: list[Any] | None = None

    def replay_broadcast(data, rank, _group, *, src):
        nonlocal broadcast_payload
        if rank == src:
            broadcast_payload = deepcopy(data)
        assert broadcast_payload is not None
        return deepcopy(broadcast_payload)

    monkeypatch.setattr(scheduler_module, "broadcast_pyobj", replay_broadcast)

    leader = _scheduler(
        _Backend(),
        tp_rank=0,
        tp_size=2,
        max_running_requests=2,
        max_batch_wait_ms=20,
    )
    follower = _scheduler(
        _Backend(),
        tp_rank=1,
        tp_size=2,
        max_running_requests=2,
        max_batch_wait_ms=20,
    )
    assert not leader.requires_tp_work_fanout

    leader.inbox.put(_message("first"))
    leader._drain_messages(block=False)
    follower._drain_messages(block=False)
    assert (
        [request.request_id for request in leader._waiting]
        == [request.request_id for request in follower._waiting]
        == ["first"]
    )

    leader.inbox.put(_message("late"))
    leader._wait_for_initial_batch()
    follower._wait_for_initial_batch()
    assert (
        [request.request_id for request in leader._waiting]
        == [request.request_id for request in follower._waiting]
        == ["first", "late"]
    )

    leader._emit_result("first", "leader-result")
    leader._emit_error("late", RuntimeError("leader-error"))
    follower._emit_result("first", "follower-result")
    follower._emit_error("late", RuntimeError("follower-error"))
    assert [leader.outbox.get_nowait().type for _ in range(2)] == ["result", "error"]
    assert follower.outbox.empty()


def test_scheduler_applies_batch_cost_without_starving_large_requests() -> None:
    backend = _Backend()
    scheduler = _scheduler(
        backend,
        max_batch_size=3,
        max_running_requests=3,
        max_batch_wait_ms=20,
        request_cost_fn=lambda payload: payload["cost"],
        max_batch_cost=3,
    )

    outputs = run_scheduler(
        scheduler,
        [
            _message("a", cost=2),
            _message("b", cost=2),
            _message("large", cost=5),
        ],
        output_count=3,
    )

    assert backend.step_batches == [("a",), ("b",), ("large",)]
    assert all(output.type == "result" for output in outputs)


def test_prepare_error_is_isolated_from_other_requests() -> None:
    backend = _Backend()
    scheduler = _scheduler(
        backend,
        max_batch_size=2,
        max_running_requests=2,
        max_batch_wait_ms=20,
    )

    outputs = run_scheduler(
        scheduler,
        [_message("bad", prepare_error=True), _message("good")],
        output_count=2,
    )
    by_id = {output.request_id: output for output in outputs}

    assert by_id["bad"].type == "error"
    assert isinstance(by_id["bad"].data, RuntimeError)
    assert by_id["good"].type == "result"
    assert backend.released == ["good"]


class _FailFirstStepBackend(_Backend):
    def step(self, states: list[_State]) -> None:
        self.step_batches.append(tuple(state.name for state in states))
        if len(self.step_batches) == 1:
            raise RuntimeError("model step failed")
        for state in states:
            state.steps += 1


def test_batch_step_error_releases_batch_and_scheduler_keeps_serving() -> None:
    backend = _FailFirstStepBackend()
    scheduler = _scheduler(
        backend,
        max_batch_size=2,
        max_running_requests=2,
        max_batch_wait_ms=20,
    )

    outputs = run_scheduler(
        scheduler,
        [_message("bad-a"), _message("bad-b"), _message("next")],
        output_count=3,
    )
    by_id = {output.request_id: output for output in outputs}

    assert backend.step_batches == [("bad-a", "bad-b"), ("next",)]
    assert by_id["bad-a"].type == "error"
    assert by_id["bad-b"].type == "error"
    assert by_id["next"].type == "result"
    assert sorted(backend.released) == ["bad-a", "bad-b", "next"]


class _BlockingBackend(_Backend):
    def __init__(self) -> None:
        super().__init__()
        self.step_started = threading.Event()
        self.allow_step = threading.Event()
        self.state_released = threading.Event()

    def step(self, states: list[_State]) -> None:
        self.step_started.set()
        if not self.allow_step.wait(timeout=2.0):
            raise TimeoutError("test did not release diffusion step")
        super().step(states)

    def release(self, state: _State) -> None:
        super().release(state)
        self.state_released.set()


def test_abort_during_model_step_suppresses_result_and_releases_state() -> None:
    backend = _BlockingBackend()
    scheduler = _scheduler(backend)
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    try:
        scheduler.inbox.put(_message("abort-me"))
        assert backend.step_started.wait(timeout=2.0)
        scheduler.abort("abort-me")
        backend.allow_step.set()
        assert backend.state_released.wait(timeout=2.0)
        try:
            output = scheduler.outbox.get(timeout=0.05)
        except queue.Empty:
            output = None
        assert output is None
        assert backend.released == ["abort-me"]
    finally:
        backend.allow_step.set()
        scheduler.stop()
        thread.join(timeout=2.0)
