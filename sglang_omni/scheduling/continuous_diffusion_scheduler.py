# SPDX-License-Identifier: Apache-2.0
"""Stage-facing scheduler for continuous diffusion requests."""

from __future__ import annotations

import collections
import logging
import queue as _queue_mod
import threading
import time
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

StateT = TypeVar("StateT")

_ABORTED_REQUEST_ID_LIMIT = 10000
_ABORTED_REQUEST_ID_RETAINED = 5000


class ContinuousDiffusionBackend(Protocol[StateT]):
    """Model-owned operations required by the generic diffusion scheduler."""

    def prepare(self, payload: Any) -> StateT: ...

    def batch_key(self, state: StateT) -> Hashable: ...

    def step(self, states: list[StateT]) -> None: ...

    def is_finished(self, state: StateT) -> bool: ...

    def finalize(self, state: StateT) -> Any: ...

    def release(self, state: StateT) -> None: ...


@dataclass
class _PendingRequest:
    request_id: str
    payload: Any
    cost: int


@dataclass
class _ResidentRequest(Generic[StateT]):
    request_id: str
    state: StateT
    batch_key: Hashable
    cost: int


class ContinuousDiffusionScheduler(Generic[StateT]):
    """Interleave request-local denoising chains one model step at a time.

    The backend owns all model and algorithm state. This scheduler only owns
    request admission, compatible batching, lifecycle, and cleanup, while
    exposing the standard Omni stage contract: ``inbox``, ``outbox``,
    ``start()``, ``stop()``, and ``abort(request_id)``.
    """

    def __init__(
        self,
        backend: ContinuousDiffusionBackend[StateT],
        *,
        max_batch_size: int = 1,
        max_running_requests: int | None = None,
        max_batch_wait_ms: int = 0,
        request_cost_fn: Callable[[Any], int] | None = None,
        max_batch_cost: int | None = None,
    ) -> None:
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self.requires_tp_work_fanout: bool = True

        self._backend = backend
        self._max_batch_size = max(int(max_batch_size), 1)
        self._max_running_requests = max(
            int(
                self._max_batch_size
                if max_running_requests is None
                else max_running_requests
            ),
            1,
        )
        self._max_batch_wait_s = max(float(max_batch_wait_ms), 0.0) / 1000.0
        self._request_cost_fn = request_cost_fn
        self._max_batch_cost = (
            max(int(max_batch_cost), 0) if max_batch_cost is not None else None
        )

        self._waiting: collections.deque[_PendingRequest] = collections.deque()
        self._waiting_ids: set[str] = set()
        self._resident: dict[str, _ResidentRequest[StateT]] = {}
        self._ready: collections.deque[str] = collections.deque()

        self._running = False
        self._aborted: set[str] = set()
        self._abort_lock = threading.Lock()

    def start(self) -> None:
        self._running = True
        try:
            while self._running:
                idle = not self._waiting and not self._resident
                self._drain_messages(block=idle)
                self._purge_aborted()
                if not self._running:
                    break

                if not self._resident and len(self._waiting) == 1:
                    self._wait_for_initial_batch()
                    self._purge_aborted()

                self._admit_waiting_requests()
                batch = self._select_batch()
                if not batch:
                    continue
                self._run_step(batch)
        finally:
            self._release_all()

    def event_loop(self) -> None:
        self.start()

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        with self._abort_lock:
            self._aborted.add(request_id)
            if len(self._aborted) <= _ABORTED_REQUEST_ID_LIMIT:
                return
            excess = len(self._aborted) - _ABORTED_REQUEST_ID_RETAINED
            for stale_request_id in list(self._aborted)[:excess]:
                self._aborted.discard(stale_request_id)

    def _is_aborted(self, request_id: str) -> bool:
        with self._abort_lock:
            return request_id in self._aborted

    def _clear_aborted(self, request_id: str) -> None:
        with self._abort_lock:
            self._aborted.discard(request_id)

    def _drain_messages(self, *, block: bool) -> None:
        if block:
            try:
                self._handle_message(self.inbox.get(timeout=0.05))
            except _queue_mod.Empty:
                return

        while True:
            try:
                message = self.inbox.get_nowait()
            except _queue_mod.Empty:
                return
            self._handle_message(message)

    def _handle_message(self, message: IncomingMessage) -> None:
        if message.type != "new_request":
            logger.warning(
                "%s: unhandled message type %r for request %s",
                self.__class__.__name__,
                message.type,
                message.request_id,
            )
            return
        if self._is_aborted(message.request_id):
            self._clear_aborted(message.request_id)
            return
        if (
            message.request_id in self._waiting_ids
            or message.request_id in self._resident
        ):
            self._emit_error(
                message.request_id,
                ValueError(f"duplicate request id: {message.request_id}"),
            )
            return

        try:
            cost = (
                max(int(self._request_cost_fn(message.data)), 0)
                if self._request_cost_fn is not None
                else 0
            )
        except Exception as exc:
            self._emit_error(message.request_id, exc)
            return

        self._waiting.append(
            _PendingRequest(
                request_id=message.request_id,
                payload=message.data,
                cost=cost,
            )
        )
        self._waiting_ids.add(message.request_id)

    def _wait_for_initial_batch(self) -> None:
        if self._max_batch_wait_s <= 0:
            return
        deadline = time.monotonic() + self._max_batch_wait_s
        while len(self._waiting) < self._max_running_requests:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                message = self.inbox.get(timeout=remaining)
            except _queue_mod.Empty:
                return
            self._handle_message(message)

    def _purge_aborted(self) -> None:
        if self._waiting:
            kept: collections.deque[_PendingRequest] = collections.deque()
            for request in self._waiting:
                if self._is_aborted(request.request_id):
                    self._waiting_ids.discard(request.request_id)
                    self._clear_aborted(request.request_id)
                else:
                    kept.append(request)
            self._waiting = kept

        for request_id, request in list(self._resident.items()):
            if not self._is_aborted(request_id):
                continue
            self._resident.pop(request_id, None)
            self._release(request)
            self._clear_aborted(request_id)

        if self._ready:
            self._ready = collections.deque(
                request_id for request_id in self._ready if request_id in self._resident
            )

    def _admit_waiting_requests(self) -> None:
        while (
            self._waiting
            and len(self._resident) < self._max_running_requests
            and self._running
        ):
            pending = self._waiting.popleft()
            self._waiting_ids.discard(pending.request_id)
            if self._is_aborted(pending.request_id):
                self._clear_aborted(pending.request_id)
                continue

            state_created = False
            try:
                state = self._backend.prepare(pending.payload)
                state_created = True
                batch_key = self._backend.batch_key(state)
                hash(batch_key)
            except Exception as exc:
                if state_created:
                    self._release_state(pending.request_id, state)
                if self._is_aborted(pending.request_id):
                    self._clear_aborted(pending.request_id)
                else:
                    self._emit_error(pending.request_id, exc)
                continue

            request = _ResidentRequest(
                request_id=pending.request_id,
                state=state,
                batch_key=batch_key,
                cost=pending.cost,
            )
            if self._is_aborted(pending.request_id):
                self._release(request)
                self._clear_aborted(pending.request_id)
                continue
            self._resident[pending.request_id] = request
            self._ready.append(pending.request_id)

    def _select_batch(self) -> list[_ResidentRequest[StateT]]:
        while self._ready and self._ready[0] not in self._resident:
            self._ready.popleft()
        if not self._ready:
            return []

        first = self._resident[self._ready.popleft()]
        selected = [first]
        batch_cost = first.cost
        candidates = len(self._ready)

        for _ in range(candidates):
            if len(selected) >= self._max_batch_size:
                break
            request_id = self._ready.popleft()
            request = self._resident.get(request_id)
            if request is None:
                continue
            compatible = request.batch_key == first.batch_key
            within_budget = (
                self._max_batch_cost is None
                or batch_cost + request.cost <= self._max_batch_cost
            )
            if compatible and within_budget:
                selected.append(request)
                batch_cost += request.cost
            else:
                self._ready.append(request_id)

        return selected

    def _run_step(self, batch: list[_ResidentRequest[StateT]]) -> None:
        try:
            self._backend.step([request.state for request in batch])
        except Exception as exc:
            logger.exception("Continuous diffusion backend step failed")
            for request in batch:
                if self._resident.pop(request.request_id, None) is None:
                    continue
                if not self._is_aborted(request.request_id):
                    self._emit_error(request.request_id, exc)
                self._release(request)
                self._clear_aborted(request.request_id)
            return

        # abort() can run from the stage event-loop thread while the model step
        # is executing in this scheduler thread.
        self._purge_aborted()
        for request in batch:
            if self._resident.get(request.request_id) is not request:
                continue
            try:
                finished = self._backend.is_finished(request.state)
            except Exception as exc:
                self._fail_request(request, exc)
                continue

            if not finished:
                self._ready.append(request.request_id)
                continue

            try:
                result = self._backend.finalize(request.state)
            except Exception as exc:
                self._fail_request(request, exc)
                continue

            self._resident.pop(request.request_id, None)
            if not self._is_aborted(request.request_id):
                self.outbox.put(
                    OutgoingMessage(
                        request_id=request.request_id,
                        type="result",
                        data=result,
                    )
                )
            self._release(request)
            self._clear_aborted(request.request_id)

    def _fail_request(
        self, request: _ResidentRequest[StateT], error: BaseException
    ) -> None:
        self._resident.pop(request.request_id, None)
        if not self._is_aborted(request.request_id):
            self._emit_error(request.request_id, error)
        self._release(request)
        self._clear_aborted(request.request_id)

    def _emit_error(self, request_id: str, error: BaseException) -> None:
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="error",
                data=error,
            )
        )

    def _release(self, request: _ResidentRequest[StateT]) -> None:
        self._release_state(request.request_id, request.state)

    def _release_state(self, request_id: str, state: StateT) -> None:
        try:
            self._backend.release(state)
        except Exception:
            logger.exception(
                "%s: cleanup failed for request %s",
                self.__class__.__name__,
                request_id,
            )

    def _release_all(self) -> None:
        for request in list(self._resident.values()):
            self._release(request)
        self._resident.clear()
        self._ready.clear()
        self._waiting.clear()
        self._waiting_ids.clear()


__all__ = ["ContinuousDiffusionBackend", "ContinuousDiffusionScheduler"]
