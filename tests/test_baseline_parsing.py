"""The two numbers the baseline reads out of llama.cpp's own report.

Everything else in the harness is measured by us; these two are taken from
the runtime's text, which makes them the place a wrong figure could enter
without anything failing.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

spec = importlib.util.spec_from_file_location(
    "baseline", os.path.join(os.path.dirname(__file__), "..", "benchmarks", "baseline.py"))
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)

REAL = """
llama_perf_context_print:        load time =   34521.19 ms
llama_perf_context_print: prompt eval time =     812.44 ms /    11 tokens (   73.86 ms per token,    13.54 tokens per second)
llama_perf_context_print:        eval time =    1234.56 ms /    64 runs   (   19.29 ms per token,    51.84 tokens per second)
llama_perf_context_print:       total time =   36568.19 ms /    75 tokens
"""


def test_generation_throughput_is_taken_not_prompt_throughput():
    """Both lines say 'tokens per second'; the prompt line is the wrong one."""
    assert baseline.throughput(REAL) == 51.84


def test_load_time_is_reported_in_seconds():
    assert baseline.load_seconds(REAL) == pytest.approx(34.52119)


def test_a_run_that_never_generated_reports_nothing_rather_than_zero():
    assert baseline.throughput("model loaded\nkilled\n") is None
    assert baseline.load_seconds("model loaded\nkilled\n") is None


def test_the_last_generation_wins_when_a_run_reports_several():
    text = REAL + REAL.replace("51.84", "12.50")
    assert baseline.throughput(text) == 12.50


def test_the_command_pins_cpu_only_execution():
    argv = baseline.argv_for(__import__("pathlib").Path("/bin/llama"),
                             __import__("pathlib").Path("/m.gguf"), 64, 16, "hi")
    assert "-ngl" in argv and argv[argv.index("-ngl") + 1] == "0", \
        "the baseline must not offload to GPU, or it measures VRAM not NVMe"
    assert "--no-warmup" in argv, "a warmup would populate the cache before the measurement"
