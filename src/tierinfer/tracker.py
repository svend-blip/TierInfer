"""What the model actually asked for, so policy can stop guessing.

A cache that knows only recency treats an expert used once and an expert used
every other token as the same thing until one of them ages out. For MoE
inference that is the wrong shape: routing is skewed, the skew is stable over
a prompt, and it is measurable while the model runs.

This module records activations and derives the signals the cache and the
predictor need. It stores counts and a bounded history, not the weights.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

#: An expert is identified by its layer and its index within that layer.
ExpertKey = tuple[int, int]


@dataclass
class ExpertStats:
    """What is known about one expert's use."""

    activations: int = 0
    last_token: int = -1
    first_token: int = -1
    reuse_distances: deque[int] = field(default_factory=lambda: deque(maxlen=32))
    loads: int = 0
    load_seconds: float = 0.0

    @property
    def mean_reuse_distance(self) -> float:
        """Tokens between uses, averaged over recent reuses.

        Small means the expert comes back soon and is worth keeping. No
        history means it has been seen once; that is not the same as "comes
        back never", so callers must treat the absence as unknown rather than
        as a large distance.
        """
        return sum(self.reuse_distances) / len(self.reuse_distances) if self.reuse_distances else 0.0

    @property
    def mean_load_seconds(self) -> float:
        return self.load_seconds / self.loads if self.loads else 0.0


class ExpertTracker:
    """Activation history over a bounded window of tokens."""

    def __init__(self, window: int = 256):
        if window < 1:
            raise ValueError("window must be at least one token")
        self.window = window
        self.token = -1
        self.stats: dict[ExpertKey, ExpertStats] = {}
        #: One entry per token: the experts that token routed to.
        self.history: deque[tuple[int, frozenset[ExpertKey]]] = deque(maxlen=window)

    # -- recording ------------------------------------------------------

    def begin_token(self) -> int:
        self.token += 1
        return self.token

    def record(self, experts: list[ExpertKey] | set[ExpertKey]) -> None:
        """Record the experts one token routed to."""
        chosen = frozenset(experts)
        for key in chosen:
            s = self.stats.get(key)
            if s is None:
                s = self.stats[key] = ExpertStats(first_token=self.token)
            elif s.last_token >= 0:
                s.reuse_distances.append(self.token - s.last_token)
            s.activations += 1
            s.last_token = self.token
        self.history.append((self.token, chosen))

    def record_load(self, key: ExpertKey, seconds: float) -> None:
        """Record what it cost to bring this expert in, for cost-aware policy."""
        s = self.stats.setdefault(key, ExpertStats())
        s.loads += 1
        s.load_seconds += seconds

    # -- signals --------------------------------------------------------

    def activation_rate(self, key: ExpertKey) -> float:
        """Share of the tokens in the window that used this expert, 0..1."""
        if not self.history:
            return 0.0
        seen = sum(1 for _, experts in self.history if key in experts)
        return seen / len(self.history)

    def recency(self, key: ExpertKey) -> int:
        """Tokens since this expert was last used; ``-1`` when never seen."""
        s = self.stats.get(key)
        return self.token - s.last_token if s and s.last_token >= 0 else -1

    def hot(self, n: int) -> list[ExpertKey]:
        """The ``n`` experts with the highest activation rate, ties by recency."""
        return sorted(
            self.stats,
            key=lambda k: (-self.activation_rate(k), self.recency(k) if self.recency(k) >= 0 else 1 << 30),
        )[:n]

    def last_token_experts(self) -> frozenset[ExpertKey]:
        return self.history[-1][1] if self.history else frozenset()

    @property
    def tokens_seen(self) -> int:
        return len(self.history)

    def coverage(self, keys: set[ExpertKey]) -> float:
        """Share of the last token's experts that ``keys`` contains.

        The number a predictor is graded on: what fraction of what was
        actually needed had already been named.
        """
        needed = self.last_token_experts()
        if not needed:
            return 0.0
        return len(needed & keys) / len(needed)
