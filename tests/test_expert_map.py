"""The seam between the Python index and the C++ assistant.

GGUF parsing lives in Python, where a wrong offset fails a test. The C++ side
reads a flat table it cannot misinterpret. That only holds if the table says
what the reader expects, so this pins the format from the writing side —
the reader's parse is `sscanf(line, "%d %d %lld %lld", ...)` and anything
else in a data line makes it refuse the whole map.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

HERE = os.path.dirname(__file__)
SCRIPT = os.path.join(HERE, "..", "tools", "trace", "expert_map.py")


def test_the_writer_exists_and_is_runnable():
    assert os.path.exists(SCRIPT)
    r = subprocess.run([sys.executable, SCRIPT, "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "expert" in r.stdout.lower()


def test_a_data_line_is_exactly_four_integers():
    """The C++ reader takes four with sscanf and refuses the map otherwise."""
    line = "3 17 1604932128 3063808"
    parts = line.split()
    assert len(parts) == 4
    assert all(p.lstrip("-").isdigit() for p in parts)


def test_the_header_lines_are_comments_the_reader_skips():
    for line in ("# tierinfer-expert-map 1", "# model x.gguf", "# layers 46 experts 128"):
        assert line.startswith("#"), "the reader skips only '#' and blank lines"


@pytest.mark.skipif(not os.path.exists(os.path.join(HERE, "..", "traces", "glm45air.expertmap")),
                    reason="no expert map has been generated on this host")
def test_the_generated_map_parses_the_way_the_reader_parses_it():
    path = os.path.join(HERE, "..", "traces", "glm45air.expertmap")
    layers, experts, ranges = set(), set(), 0
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            assert len(parts) == 4, f"line {lineno} has {len(parts)} fields, reader wants 4"
            layer, expert, off, nbytes = (int(p) for p in parts)
            assert off >= 0 and nbytes > 0, f"line {lineno} has a bad range"
            layers.add(layer)
            experts.add(expert)
            ranges += 1
    assert ranges > 0
    assert len(experts) == 128, f"expected 128 experts, found {len(experts)}"
    assert ranges == len(layers) * len(experts) * 3, \
        "every expert should contribute gate, up and down"
