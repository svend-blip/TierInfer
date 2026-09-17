"""Starting expert reads before the token that needs them, and counting the cost.

The pieces exist separately: a predictor that ranks, a streamer that reads
without blocking, a cache that decides what to keep. This is where they meet,
and it is the only place in the project where a guess is allowed to cause
I/O.

The rule that makes that safe is the one SCOPE goal 15 states and goal 8
repeats: a prediction decides what to *have ready*, never what to *use*. When
routing arrives and names an expert nobody prefetched, this issues an exact
synchronous load for it. Always, without consulting anything. A prefetcher
that could not do that would be a correctness risk dressed as a speedup.

Four numbers come out, and they are the four SCOPE goal 7 asks for:

``stalls_avoided``    experts that were routed to and already resident or
                      in flight — the work the prefetcher actually saved
``stalls``            experts that were routed to and had to be fetched then
                      and there, which is what it failed to save
``accuracy``          of everything prefetched, the share that got used
``lead_seconds``      how long before it was needed each useful prefetch was
                      issued — a prefetch that lands one millisecond early
                      saved almost nothing, and counting it the same as one
                      that landed a second early would flatter the policy
``wasted_bytes``      read for experts that were never routed to

Waste is not a defect on its own. Holding 16 of 128 experts ready to catch 8
means at least half of what is read goes unused by construction; the question
is whether the stalls avoided were worth the bandwidth. Both are reported so
that trade is visible rather than asserted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from .cache import ExpertCache
from .index import ByteRange
from .predict import Predictor
from .stream import ExpertStreamer, Load, PoolExhausted, State, StreamError
from .tracker import ExpertKey, ExpertTracker


@dataclass
class PrefetchStats:
    issued: int = 0
    used: int = 0
    cancelled: int = 0
    stalls: int = 0
    stalls_avoided: int = 0
    bytes_issued: int = 0
    wasted_bytes: int = 0
    lead_total: float = 0.0
    lead_count: int = 0
    pool_exhausted: int = 0
    exact_fallbacks: int = 0

    @property
    def accuracy(self) -> float:
        """Of what was prefetched, the share that was routed to."""
        return self.used / self.issued if self.issued else 0.0

    @property
    def stall_rate(self) -> float:
        total = self.stalls + self.stalls_avoided
        return self.stalls / total if total else 0.0

    @property
    def mean_lead_seconds(self) -> float:
        return self.lead_total / self.lead_count if self.lead_count else 0.0


@dataclass
class _InFlight:
    load: Load
    issued_at: float
    nbytes: int


class Prefetcher:
    """Issues speculative reads, and is honest about what they bought.

    ``ranges_for`` is how it gets from an expert key to bytes — normally
    ``ModelIndex.expert``. It is injected rather than assumed so this can be
    tested and so a different layout does not need a different prefetcher.
    """

    def __init__(self, streamer: ExpertStreamer, cache: ExpertCache,
                 predictor: Predictor,
                 ranges_for: Callable[[ExpertKey], Sequence[ByteRange]],
                 *, tracker: ExpertTracker | None = None, depth: int = 16) -> None:
        self.streamer = streamer
        self.cache = cache
        self.predictor = predictor
        self.ranges_for = ranges_for
        self.tracker = tracker
        self.depth = depth
        self.stats = PrefetchStats()
        self._inflight: dict[ExpertKey, _InFlight] = {}

    # -- speculation ----------------------------------------------------

    def before_layer(self, layer: int, sofar: Mapping[int, Sequence[int]]) -> list[ExpertKey]:
        """Start reads for what this layer is likely to want. Returns what was issued.

        Anything already resident or already in flight is skipped — a
        prefetcher that re-reads what it has is measuring its own noise.
        The pool running out is a normal outcome at depth, not an error: it
        is counted and speculation stops there for this layer.
        """
        prediction = self.predictor.predict(layer, dict(sofar))
        issued: list[ExpertKey] = []
        confidence = prediction.as_scores()
        for expert in prediction.top(self.depth):
            key = (layer, expert)
            if key in self.cache or key in self._inflight:
                continue
            ranges = list(self.ranges_for(key))
            if not ranges:
                continue
            try:
                load = self.streamer.submit(key, ranges)
            except PoolExhausted:
                self.stats.pool_exhausted += 1
                break
            nbytes = sum(r.nbytes for r in ranges)
            self._inflight[key] = _InFlight(load=load, issued_at=time.perf_counter(),
                                            nbytes=nbytes)
            self.stats.issued += 1
            self.stats.bytes_issued += nbytes
            issued.append(key)
        if confidence:
            self.cache.set_confidence({(layer, e): c for e, c in confidence.items()})
        return issued

    # -- what actually happened -----------------------------------------

    def on_routing(self, layer: int, experts: Sequence[int]) -> dict[ExpertKey, bytes | None]:
        """Settle a layer against its real routing.

        Every routed expert comes back present: from the cache, from a
        prefetch that is now waited on, or — when nothing anticipated it —
        from an exact synchronous read. The third case is a stall, and it is
        counted as one.
        """
        out: dict[ExpertKey, bytes | None] = {}
        now = time.perf_counter()
        for expert in experts:
            key = (layer, expert)
            if key in self.cache:
                self.stats.stalls_avoided += 1
                out[key] = self.cache.get(key)
                continue
            flight = self._inflight.pop(key, None)
            if flight is not None:
                out[key] = self._collect(key, flight, now)
                continue
            # Nobody saw this coming. Read it exactly, consulting nothing.
            self.stats.stalls += 1
            self.stats.exact_fallbacks += 1
            data = self.streamer.load_now(list(self.ranges_for(key)))
            self._admit(key, data)
            out[key] = data
        return out

    def _collect(self, key: ExpertKey, flight: _InFlight, now: float) -> bytes | None:
        """Take delivery of a prefetch that was right."""
        try:
            view = self.streamer.wait(flight.load, timeout=120)
        except StreamError:
            # The speculative read failed. That is not a correctness problem:
            # fall back to the exact path, and count it as a stall, because
            # from the token's point of view that is exactly what it was.
            self.stats.stalls += 1
            self.stats.exact_fallbacks += 1
            data = self.streamer.load_now(list(self.ranges_for(key)))
            self._admit(key, data)
            return data
        self.stats.used += 1
        self.stats.stalls_avoided += 1
        self.stats.lead_total += max(0.0, now - flight.issued_at)
        self.stats.lead_count += 1
        data = bytes(view)
        self.streamer.release(flight.load)
        self._admit(key, data)
        return data

    def _admit(self, key: ExpertKey, data: bytes) -> None:
        if self.tracker:
            self.tracker.record({key})
        self.cache.put(key, data, len(data))

    def drop_unused(self) -> int:
        """Give up on everything still in flight. Returns how many.

        Called at a token boundary: a prefetch that the token did not want is
        wrong about *this* token, and holding its slot starves the next one.
        """
        dropped = 0
        for key, flight in list(self._inflight.items()):
            self._inflight.pop(key, None)
            self.stats.wasted_bytes += flight.nbytes
            self.stats.cancelled += 1
            if not self.streamer.cancel(flight.load):
                # Already read: take delivery only to give the buffer back.
                try:
                    self.streamer.wait(flight.load, timeout=120)
                except StreamError:
                    pass
                if flight.load.state in (State.READY, State.FAILED):
                    try:
                        self.streamer.release(flight.load)
                    except StreamError:
                        pass
            dropped += 1
        return dropped

    def end_token(self) -> None:
        """Close the token: drop what went unused, and start the next window."""
        self.drop_unused()
        if self.tracker:
            self.tracker.begin_token()
