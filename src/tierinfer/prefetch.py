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

and one more that the first version folded into ``used``:

``late``              prefetches that were routed to *before they had
                      finished* — demand waited on them. A late prefetch is
                      not a stall (the read was already in flight) and not a
                      useful one (the token still waited); it is its own
                      thing, and ``late_wait_seconds`` is what it cost.

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
    late: int = 0
    late_wait_seconds: float = 0.0
    timed_out: int = 0
    unmappable: int = 0

    @property
    def useful(self) -> int:
        """Prefetches that had landed before the token asked for them."""
        return self.used - self.late

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
                 *, tracker: ExpertTracker | None = None, depth: int = 16,
                 wait_timeout: float = 120.0) -> None:
        self.streamer = streamer
        self.wait_timeout = wait_timeout
        self.cache = cache
        self.predictor = predictor
        self.ranges_for = ranges_for
        self.tracker = tracker
        self.depth = depth
        self.stats = PrefetchStats()
        self._inflight: dict[ExpertKey, _InFlight] = {}
        #: Loads given up on while still reading. Their buffers come back
        #: when they finish, reaped at the next token boundary — never waited on.
        self._orphans: list[Load] = []
        #: What this token has routed to so far, recorded once at ``end_token``.
        self._routed: set[ExpertKey] = set()

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
            try:
                ranges = list(self.ranges_for(key))
            except Exception:                   # noqa: BLE001 — a guess about an expert that cannot be addressed
                # Speculation may name an expert the index cannot resolve.
                # That is a wrong guess, not a failure: skip it and count it.
                # The same lookup on the *routed* path raises, as it must.
                self.stats.unmappable += 1
                continue
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
            self._routed.add(key)
            if key in self.cache:
                self.stats.stalls_avoided += 1
                out[key] = self.cache.get(key)
                continue
            flight = self._inflight.pop(key, None)
            if flight is not None:
                out[key] = self._collect(key, flight, now)
                continue
            # Nobody saw this coming. Read it exactly, consulting nothing.
            out[key] = self._exact(key)
        return out

    def _collect(self, key: ExpertKey, flight: _InFlight, now: float) -> bytes | None:
        """Take delivery of a prefetch that was right.

        Right, but not necessarily in time: a load still reading when the
        routing names it is *late*, and the wait is measured and counted
        apart from the ones that had landed.
        """
        landed = flight.load.state is State.READY
        try:
            view = self.streamer.wait(flight.load, timeout=self.wait_timeout)
        except TimeoutError:
            # Still reading after the whole timeout. The buffer belongs to the
            # worker until it finishes, so it cannot be released here; it is
            # reaped at a token boundary. The token gets its bytes exactly.
            self.stats.timed_out += 1
            self._orphans.append(flight.load)
            return self._exact(key)
        except StreamError:
            # The speculative read failed. That is not a correctness problem:
            # fall back to the exact path, and count it as a stall, because
            # from the token's point of view that is exactly what it was.
            return self._exact(key)
        waited = time.perf_counter() - now
        self.stats.used += 1
        self.stats.stalls_avoided += 1
        self.stats.lead_total += max(0.0, now - flight.issued_at)
        self.stats.lead_count += 1
        if not landed:
            self.stats.late += 1
            self.stats.late_wait_seconds += waited
        data = bytes(view)
        self._admit(key, data, flight.load.read_seconds)
        self.streamer.release(flight.load)
        return data

    def _exact(self, key: ExpertKey) -> bytes:
        """The path that consults nothing, counted as the stall it is."""
        self.stats.stalls += 1
        self.stats.exact_fallbacks += 1
        t0 = time.perf_counter()
        data = self.streamer.load_now(list(self.ranges_for(key)))
        self._admit(key, data, time.perf_counter() - t0)
        return data

    def _admit(self, key: ExpertKey, data: bytes, load_seconds: float = 0.0) -> None:
        # The cache's value function has a reload-cost term (SCOPE 7.3,
        # "transfer cost"); until this call existed nothing fed it and the
        # term was a constant. The streamer's own per-load read time is the
        # measurement.
        if self.tracker and load_seconds > 0:
            self.tracker.record_load(key, load_seconds)
        self.cache.put(key, data, len(data))

    def drop_unused(self) -> int:
        """Give up on everything still in flight. Returns how many.

        Called at a token boundary: a prefetch that the token did not want is
        wrong about *this* token, and holding its slot starves the next one.

        Never waits. A load that has not started is cancelled; one that has
        finished gives its buffer back now; one that is mid-read is orphaned
        and its buffer is reaped the next time this runs. The first version
        waited up to two minutes here for a read to finish so it could be
        thrown away, which put speculation on the token's critical path.
        """
        dropped = 0
        for key, flight in list(self._inflight.items()):
            self._inflight.pop(key, None)
            self.stats.wasted_bytes += flight.nbytes
            self.stats.cancelled += 1
            if not self.streamer.cancel(flight.load):
                self._orphans.append(flight.load)
            dropped += 1
        self._reap()
        return dropped

    def _reap(self) -> None:
        """Return the buffers of orphaned loads that have since finished."""
        still: list[Load] = []
        for load in self._orphans:
            if load.state in (State.READY, State.FAILED, State.CANCELLED):
                try:
                    self.streamer.release(load)
                except StreamError:
                    pass
            elif load.state is State.RELEASED:
                pass
            else:
                still.append(load)
        self._orphans = still

    @property
    def orphans(self) -> int:
        """Loads given up on that are still holding a buffer."""
        return len(self._orphans)

    def end_token(self) -> None:
        """Close the token: record what it routed to, drop what went unused,
        and start the next window.

        The tracker is told about the token *once*, with everything it routed
        to. The first version recorded each expert as it was admitted, which
        made the tracker's per-token history a per-admission history and its
        activation rate meaningless.
        """
        self.drop_unused()
        if self.tracker:
            if self._routed:
                self.tracker.record(self._routed)
            self.tracker.begin_token()
        self._routed = set()
