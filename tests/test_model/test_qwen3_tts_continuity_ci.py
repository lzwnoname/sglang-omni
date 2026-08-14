# SPDX-License-Identifier: Apache-2.0
"""Playback-continuity regression gate for Qwen3-TTS streaming (#1399 §3.3).

Qwen3-TTS Base buffers ``initial_codec_chunk_frames`` codec frames before it
emits the first vocoder chunk. When that buffer was 1 frame (~80 ms) an
immediate-play client underran on the very first seam under concurrency: the
first follow-up chunk arrived hundreds of milliseconds after the first payload
had finished playing.

This test pins the fix as a contract rather than as a number that happens to be
green. The controlled A/B behind it, at c32 on one physical GPU, was::

    init=1     C50 =  46.88%   (reproduces the old failure)
    default=8  C50 = 100.00%   (shipped default)

so the gate demonstrably catches the regression it is meant to catch.

The contract is deliberately narrow -- continuity, plus proof that enough
requests were actually scored. TTFC and throughput are printed but not gated:
they exist so a future buffering change cannot hide a large latency or
throughput tradeoff behind a green continuity number, and belong to the speed
benchmarks rather than to §3.3's correctness contract.

Usage:
    pytest tests/test_model/test_qwen3_tts_continuity_ci.py -s -x

"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from benchmarks.benchmarker.utils import managed_omni_server
from benchmarks.dataset.prepare import DATASETS, download_dataset
from benchmarks.eval.benchmark_tts_seedtts import (
    TtsSeedttsBenchmarkConfig,
    run_tts_seedtts_benchmark,
)
from benchmarks.metrics.performance import print_speed_summary
from tests.test_model.omni_router_utils import _find_available_port_range
from tests.utils import (
    MetricCheckCollector,
    server_log_file,
    wait_for_gpu_memory_release,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Qwen3-TTS needs an explicit pipeline config (unlike Higgs / ZONOS2, which are
# auto-detected from the checkpoint).
QWEN3_TTS_MODEL_PATH = os.environ.get(
    "QWEN3_TTS_MODEL_PATH", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
)
QWEN3_TTS_MODEL_NAME = os.environ.get("QWEN3_TTS_MODEL_NAME", QWEN3_TTS_MODEL_PATH)
QWEN3_TTS_SERVER_CONFIG = os.environ.get(
    "QWEN3_TTS_SERVER_CONFIG",
    str(PROJECT_ROOT / "examples" / "configs" / "qwen3_tts_0_6b.yaml"),
)

# Single worker, c32. c32 is the load at which the old default failed; a lower
# concurrency does not reproduce it.
CONCURRENCY = 32
MAX_SAMPLES = 64

# Required gates.
C50_MIN_PASS_RATE = 90.0
MIN_SCORED_CONTINUITY_REQUESTS = 50

# First startup captures CUDA graphs for the tts_engine and can take minutes.
STARTUP_TIMEOUT = 900


def _continuity_benchmark_config(
    port: int,
    meta: str,
    output_dir: str,
) -> TtsSeedttsBenchmarkConfig:
    return TtsSeedttsBenchmarkConfig(
        model=QWEN3_TTS_MODEL_NAME,
        port=port,
        meta=meta,
        output_dir=output_dir,
        max_samples=MAX_SAMPLES,
        concurrency=CONCURRENCY,
        voice_clone=True,
        ref_format="references",
        stream=True,
        # The raw-PCM streaming client is the only path that records chunk
        # arrival times and per-chunk audio durations, which is what the
        # underrun metric is computed from. The benchmark CLI forces this when
        # --stream is passed; constructing the config directly does not, so a
        # "wav" default here would silently produce a run with no continuity
        # data at all.
        response_format="pcm",
        # Leave initial_codec_chunk_frames unset so the run exercises the
        # shipped server-side default. Overriding it here would test the
        # request-override path instead of the default this gate protects.
        initial_codec_chunk_frames=None,
    )


def _run_continuity_benchmark(port: int, meta: str, output_dir: str) -> dict:
    results = asyncio.run(
        run_tts_seedtts_benchmark(_continuity_benchmark_config(port, meta, output_dir))
    )
    assert "summary" in results, f"Missing 'summary'. Keys: {list(results.keys())}"
    return results


@pytest.fixture(scope="module")
def dataset_repo() -> str:
    # The full SeedTTS set: seedtts-50 holds 50 samples and cannot supply the
    # 64 requests this gate scores.
    repo_id = DATASETS["seedtts"]
    download_dataset(repo_id, quiet=True)
    return repo_id


@pytest.fixture(scope="module")
def continuity_summary(
    dataset_repo: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict:
    """Warm the full c32 graph, then score a second c32 pass.

    ``BenchmarkRunner._warmup`` dispatches its warmup requests sequentially, so
    ``TtsSeedttsBenchmarkConfig.warmup`` never builds a batch-32 decode no
    matter how high it is set. Only a full concurrent pass exercises the c32
    CUDA-graph shapes, so the first run is discarded and the second is scored --
    otherwise first-run graph capture lands inside the measured seams and shows
    up as underrun.
    """
    port = _find_available_port_range(1)
    log_file = server_log_file(tmp_path_factory, "qwen3_tts_continuity_server_logs")
    warmup_dir = str(tmp_path_factory.mktemp("continuity_c32_warmup"))
    scored_dir = str(tmp_path_factory.mktemp("continuity_c32_scored"))

    with managed_omni_server(
        model_path=QWEN3_TTS_MODEL_PATH,
        port=port,
        host="127.0.0.1",
        log_file=log_file,
        server_config=QWEN3_TTS_SERVER_CONFIG,
        timeout=STARTUP_TIMEOUT,
        wait_for_gpu_release=True,
    ):
        _run_continuity_benchmark(port, dataset_repo, warmup_dir)
        results = _run_continuity_benchmark(port, dataset_repo, scored_dir)
    wait_for_gpu_memory_release()
    return results["summary"]


@pytest.mark.benchmark
def test_streaming_playback_continuity_c32(continuity_summary: dict) -> None:
    print_speed_summary(
        continuity_summary,
        QWEN3_TTS_MODEL_NAME,
        CONCURRENCY,
        title="Qwen3-TTS Streaming Playback Continuity",
    )

    checks = MetricCheckCollector(f"Qwen3-TTS playback continuity c{CONCURRENCY}")

    # A missing c50 means no request carried streaming audio chunks at all, so
    # the summary was never gated. That must fail loudly instead of passing on
    # an absent key.
    checks.check(
        "c50" in continuity_summary,
        "c50 missing from the summary: the run produced no streaming audio "
        "chunks, so playback continuity was never measured.",
    )

    scored = continuity_summary.get("playback_continuity_requests")
    na_requests = continuity_summary.get("playback_continuity_na_requests")
    c50 = continuity_summary.get("c50")

    # Without this floor a change that turns most streams into single-chunk
    # (N/A) responses would report a perfect C50 over a handful of requests and
    # sail through the gate.
    checks.check(
        scored is not None and scored >= MIN_SCORED_CONTINUITY_REQUESTS,
        f"Only {scored} scored playback-continuity requests, expected at least "
        f"{MIN_SCORED_CONTINUITY_REQUESTS} of {MAX_SAMPLES} "
        f"(N/A single-chunk requests: {na_requests}).",
    )
    checks.check(
        c50 is not None and c50 >= C50_MIN_PASS_RATE,
        f"C50 was {c50}%, expected at least {C50_MIN_PASS_RATE}% at "
        f"c{CONCURRENCY} (scored={scored}, N/A={na_requests}).",
    )

    failed = continuity_summary.get("failed_requests", 0)
    checks.check(
        not failed,
        f"{failed} of {MAX_SAMPLES} requests failed; continuity numbers from a "
        "partially failed run are not meaningful.",
    )

    checks.assert_all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s", "-x"]))
