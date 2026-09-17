"""The numbers the baseline reads out of llama.cpp's own report.

Everything else in the harness is measured by us; these are taken from the
runtime's text, which makes them the place a wrong figure enters without
anything failing. Two build formats are in play, and a parser that knows only
one of them prints a table of dashes instead of raising.
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

# Captured verbatim from llama.cpp b9888-cb295bf59 on this host.
MODERN = """
> hi

|- [Start thinking]

[ Prompt: 368,9 t/s | Generation: 67,4 t/s ]

Exiting...
"""

LEGACY = """
llama_perf_context_print:        load time =   34521.19 ms
llama_perf_context_print: prompt eval time =     812.44 ms /    11 tokens (   73.86 ms per token,    13.54 tokens per second)
llama_perf_context_print:        eval time =    1234.56 ms /    64 runs   (   19.29 ms per token,    51.84 tokens per second)
llama_perf_context_print:       total time =   36568.19 ms /    75 tokens
"""


# -- the format this host actually produces -----------------------------


def test_the_current_build_reports_generation_and_prompt_separately():
    assert baseline.throughput(MODERN) == pytest.approx(67.4)
    assert baseline.prompt_throughput(MODERN) == pytest.approx(368.9)


def test_a_decimal_comma_is_read_as_a_decimal():
    """This host's locale is Danish: '67,4' parsed as a point-decimal is 67."""
    assert baseline.throughput("[ Prompt: 1,5 t/s | Generation: 67,4 t/s ]") == pytest.approx(67.4)


def test_a_decimal_point_still_works():
    assert baseline.throughput("[ Prompt: 1.5 t/s | Generation: 67.4 t/s ]") == pytest.approx(67.4)


def test_the_current_build_reports_no_load_time_and_none_is_invented():
    assert baseline.load_seconds(MODERN) is None


# -- the older format ---------------------------------------------------


def test_generation_throughput_is_taken_not_prompt_throughput():
    """Both legacy lines end in 'tokens per second'; the prompt line is wrong."""
    assert baseline.throughput(LEGACY) == 51.84


def test_legacy_load_time_is_reported_in_seconds():
    assert baseline.load_seconds(LEGACY) == pytest.approx(34.52119)


def test_the_last_generation_wins_when_a_legacy_run_reports_several():
    assert baseline.throughput(LEGACY + LEGACY.replace("51.84", "12.50")) == 12.50


# -- neither -----------------------------------------------------------


def test_a_run_that_never_generated_reports_nothing_rather_than_zero():
    for text in ("model loaded\nkilled\n", "", "Loading model... |-\\|/"):
        assert baseline.throughput(text) is None
        assert baseline.prompt_throughput(text) is None
        assert baseline.load_seconds(text) is None


# -- the command --------------------------------------------------------


def test_the_command_uses_the_flag_that_actually_terminates():
    """-no-cnv is rejected by this build ('--no-conversation is not supported
    by llama-cli') and the run then waits on a stdin a benchmark has not got.
    That cost one condition 1 127 seconds of doing nothing and a 3.9 GB log."""
    argv = baseline.argv_for(__import__("pathlib").Path("/bin/llama"),
                             __import__("pathlib").Path("/m.gguf"), 4, 16, "hi")
    assert "-st" in argv, "nothing in the command makes the run end"
    assert "-no-cnv" not in argv and "--no-conversation" not in argv


def test_the_command_pins_cpu_only_execution():
    argv = baseline.argv_for(__import__("pathlib").Path("/bin/llama"),
                             __import__("pathlib").Path("/m.gguf"), 4, 16, "hi")
    assert "-ngl" in argv and argv[argv.index("-ngl") + 1] == "0", \
        "the baseline must not offload to GPU, or it measures VRAM not NVMe"
    assert "--no-warmup" in argv, "a warmup would populate the cache before the measurement"
