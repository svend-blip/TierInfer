"""The prefetch depth adapts to what prefetch achieved (TI-POLICY-006)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from tierinfer.index import load  # noqa: E402
from tierinfer.loader import LoaderServer  # noqa: E402
from test_loader import split_model  # noqa: E402


def _server(tmp_path, depth=8):
    shards = split_model(tmp_path, count=2, layers_per=2)
    ix = load(shards[0])
    return LoaderServer(ix, ram_bytes=64 << 20, workers=1, depth=depth, adapt_depth=True, max_depth=32,
                        verbose=False, drop_page_cache=False)


def _tokens(server, n, *, issued, useful, late, wasted):
    """n tokens over which prefetch did this much (spread evenly)."""
    for i in range(n):
        server.stats.prefetch_issued += issued // n
        server.stats.prefetch_useful += useful // n
        server.stats.prefetch_late += late // n
        server.stats.prefetch_wasted += wasted // n
        server._token_keys = {(0, 0)}
        server._end_token()


def test_low_yield_halves_and_high_yield_doubles(tmp_path):
    s = _server(tmp_path, depth=8)
    try:
        _tokens(s, 8, issued=80, useful=16, late=8, wasted=56)          # 20 % yield
        assert s.depth == 4, s.depth_changes
        _tokens(s, 8, issued=80, useful=16, late=8, wasted=56)
        assert s.depth == 2
        _tokens(s, 8, issued=80, useful=64, late=8, wasted=8)           # 80 % yield, 10 % late
        assert s.depth == 4
        _tokens(s, 8, issued=80, useful=64, late=8, wasted=8)
        assert s.depth == 8
        assert [d for _, d, _ in s.depth_changes] == [4, 2, 4, 8]
    finally:
        s.close()


def test_high_yield_but_late_does_not_grow_and_few_guesses_hold(tmp_path):
    s = _server(tmp_path, depth=8)
    try:
        _tokens(s, 8, issued=80, useful=64, late=40, wasted=0)          # useful yet half late: no growth
        assert s.depth == 8
        _tokens(s, 8, issued=4, useful=0, late=0, wasted=4)             # too few guesses to judge
        assert s.depth == 8
        assert s.depth_changes == []
    finally:
        s.close()


def test_depth_zero_never_adapts(tmp_path):
    s = _server(tmp_path, depth=0)
    try:
        assert s.adapt_depth is False
        _tokens(s, 16, issued=0, useful=0, late=0, wasted=0)
        assert s.depth == 0
    finally:
        s.close()
