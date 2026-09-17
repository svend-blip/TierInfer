"""A bounded RAM cache that keeps experts by what they are worth.

**This module opened with a claim that measurement has since falsified.** It
said least-recently-used was the wrong default here: MoE routing is skewed
and the skew is stable over a prompt, so an expert used by a third of the
tokens should be worth more than one used once. Against a synthetic Zipf
trace that held, by up to 13 points of hit rate. Against 800 tokens of
routing captured out of GLM-4.5-Air (`tools/trace`), it did not. The skew is
there in the marginal, but recency is the stronger signal by a wide margin,
and a policy leading with frequency reads the weaker one:

    cache   frequency only   recency only   LRU     (trace A, measured)
     4 GB            30.1%          37.6%   37.6%
     8 GB            44.1%          51.9%   51.9%
    16 GB            64.3%          69.7%   69.7%

So the default is recency-led, and configured that way this policy *equals*
LRU rather than beating it. That is the honest position, and the terms that
could still earn their place are the ones this model cannot exercise:

    value = (w_rate·rate + w_recency·recency + w_conf·confidence)
            × reload cost ÷ size

``size`` and ``reload cost`` are inert on a model whose experts are all
9.97 MB and all cost the same to fetch. On a mixed-quantisation model, or one
tiering experts against attention weights, they are the whole point.
``confidence`` was measured too, fed from the real predictors on real
routing, and moved the hit rate by 0.1 of a point — because by the time a
prediction says an expert is likely needed, recency is already keeping it.
Prediction earns its place in the *prefetch* path, deciding what to fetch,
not in eviction, deciding what to retain.

Recency is measured in accesses, not tokens. One token touches every routed
expert of every layer — 360 of them here — so at token granularity almost
every resident entry ties with every other, and the policy cannot order what
LRU orders exactly. That version reached 43.4% against LRU's 51.9%.

**Victim selection.** An exact scan of every unpinned entry costs 37 µs at
100 residents and 1 031 µs at 3 000, which a 32 GB cache holding 3 670
experts cannot afford. A lazy min-heap of stored scores replaces it — but a
heap alone is wrong for a value that changes on every access, which recency
does. Two structures are kept: the heap, whose stored scores are revalidated
on the way out, and the access order, whose front is the exact
least-recently-used entry in O(1). Eviction compares both by true value and
takes the lower. Without the second, the heap lost 5 to 12 points against
plain LRU on real routing, silently, with `heap_fallbacks` reading zero.

Pinned entries are never scored: the shared expert and the router are needed
by every token and are not candidates.
"""

from __future__ import annotations

import heapq
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable

from .tracker import ExpertKey, ExpertTracker


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    insertions: int = 0
    evictions: int = 0
    bytes_admitted: int = 0
    bytes_evicted: int = 0
    rejected_oversize: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


@dataclass
class _Entry:
    key: ExpertKey
    nbytes: int
    payload: object
    pinned: bool = False
    inserted_at: int = 0
    last_hit: int = 0


class ExpertCache:
    """Bounded by bytes, not by count: experts are not all the same size."""

    def __init__(
        self,
        capacity_bytes: int,
        tracker: ExpertTracker | None = None,
        *,
        default_reload_seconds: float = 0.010,
        w_rate: float = 0.0,
        w_recency: float = 1.0,
        w_confidence: float = 1.0,
        recency_decay: float = 0.9,
    ):
        if capacity_bytes <= 0:
            raise ValueError("capacity must be positive")
        self.capacity_bytes = capacity_bytes
        self.tracker = tracker
        self.default_reload_seconds = default_reload_seconds
        if not 0.0 < recency_decay < 1.0:
            raise ValueError("recency_decay must be strictly between 0 and 1")
        #: How the three signals are combined. The defaults reproduce the
        #: original rate-only policy exactly, so nothing changes until a
        #: caller asks for it; `benchmarks/cache_policy.py --recency` measures
        #: the alternative.
        self.w_rate = w_rate
        self.w_recency = w_recency
        self.w_confidence = w_confidence
        self.recency_decay = recency_decay
        self.stats = CacheStats()
        self._entries: dict[ExpertKey, _Entry] = {}
        #: Access order, oldest first. The recency term of the value function
        #: changes on every hit, which a stored-score heap cannot track; this
        #: gives the exact least-recently-used entry in O(1).
        self._order: "OrderedDict[ExpertKey, None]" = OrderedDict()
        self._bytes = 0
        self._clock = 0
        self._confidence: dict[ExpertKey, float] = {}
        #: Lazy min-heap of (value, sequence, key). Entries are never removed
        #: on update, only re-pushed; stale ones are discarded on the way out.
        self._heap: list[tuple[float, int, ExpertKey]] = []
        self._seq = 0
        #: How far a stored score may drift below the true one before the
        #: candidate is re-pushed rather than evicted.
        self.revalidate_tolerance = 0.25
        #: Cap on re-pushes per eviction, so a drifting window cannot turn one
        #: eviction into an unbounded loop. Exceeding it falls back to a scan.
        self.revalidate_budget = 32
        self.heap_fallbacks = 0

    # -- state ----------------------------------------------------------

    @property
    def used_bytes(self) -> int:
        return self._bytes

    @property
    def free_bytes(self) -> int:
        return self.capacity_bytes - self._bytes

    def __contains__(self, key: ExpertKey) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def keys(self) -> list[ExpertKey]:
        return list(self._entries)

    # -- use ------------------------------------------------------------

    def get(self, key: ExpertKey):
        """Return the cached payload, or ``None``. Counts as a lookup either way."""
        self._clock += 1
        entry = self._entries.get(key)
        if entry is None:
            self.stats.misses += 1
            return None
        entry.last_hit = self._clock
        self._touch(key)
        self.stats.hits += 1
        return entry.payload

    def put(self, key: ExpertKey, payload: object, nbytes: int, *, pinned: bool = False) -> bool:
        """Admit an expert, evicting as needed. False when it cannot fit at all."""
        if nbytes > self.capacity_bytes:
            self.stats.rejected_oversize += 1
            return False
        self._clock += 1
        if key in self._entries:
            self._drop(key, evicted=False)
        while self._bytes + nbytes > self.capacity_bytes:
            victim = self._choose_victim(exclude=key)
            if victim is None:
                self.stats.rejected_oversize += 1
                return False
            self._drop(victim, evicted=True)
        self._entries[key] = _Entry(
            key=key, nbytes=nbytes, payload=payload, pinned=pinned,
            inserted_at=self._clock, last_hit=self._clock,
        )
        self._order[key] = None
        self._order.move_to_end(key, last=True)
        self._bytes += nbytes
        self.stats.insertions += 1
        self.stats.bytes_admitted += nbytes
        if not pinned:
            self._push(key)
        return True

    def pin(self, key: ExpertKey) -> None:
        """Keep this expert regardless of score. Raises when it is not resident."""
        self._entries[key].pinned = True

    def unpin(self, key: ExpertKey) -> None:
        self._entries[key].pinned = False
        self._push(key)

    def set_confidence(self, scores: dict[ExpertKey, float]) -> None:
        """A predictor's confidence for the next token, 0..1 per expert.

        Confidence raises a resident expert's value, so the heap's stored
        score for it becomes an underestimate — which the revalidation on the
        way out catches. Only the named experts are re-pushed; the rest keep
        whatever ordering they had.
        """
        self._confidence = dict(scores)
        for key in scores:
            if key in self._entries and not self._entries[key].pinned:
                self._push(key)

    # -- policy ---------------------------------------------------------

    def recency_score(self, key: ExpertKey) -> float:
        """How recently this expert was used, as a 0..1 score.

        Measured in *accesses*, not tokens. The tracker's ``recency`` counts
        whole tokens, and one token touches every routed expert of every
        layer — 360 of them on the model this was built for. At that
        granularity almost every resident expert ties with almost every
        other, and the policy cannot tell apart things LRU orders precisely.
        Measured on real routing, token-granularity recency reached 43.4%
        against LRU's 51.9%; the entry's own access clock closes that gap.
        """
        entry = self._entries.get(key)
        if entry is None:
            return 0.0
        if self._clock <= 0:
            return 1.0
        return entry.last_hit / self._clock

    def value(self, key: ExpertKey) -> float:
        """What this resident expert is worth keeping. Higher survives.

            value = (w_rate·rate + w_recency·recency + w_conf·confidence)
                    × reload cost ÷ size

        The three weights exist because the first version of this had only
        the first term, with recency demoted to a tie-break, and that ordering
        turned out to be backwards. Against a synthetic Zipf trace it beat LRU
        by up to 13 points; against 400 tokens of routing measured out of
        GLM-4.5-Air it *lost* to LRU by 6 to 9. The predictor evaluation on
        the same measured routing says why: frequency is the weakest signal in
        real routing (27.6% recall at k=8) and recency the strongest (38.3%).
        A policy weighting frequency first was reading the weaker signal.
        """
        entry = self._entries[key]
        rate = self.tracker.activation_rate(key) if self.tracker else 0.0
        confidence = self._confidence.get(key, 0.0)
        reload = self.default_reload_seconds
        if self.tracker:
            s = self.tracker.stats.get(key)
            if s and s.loads:
                reload = s.mean_load_seconds
        signal = (self.w_rate * rate
                  + self.w_recency * self.recency_score(key)
                  + self.w_confidence * confidence)
        return signal * reload / max(entry.nbytes, 1)

    def _touch(self, key: ExpertKey) -> None:
        """Move an entry to the back of the access order. O(1)."""
        self._order.move_to_end(key, last=True)

    def _lru_front(self, exclude: ExpertKey | None) -> ExpertKey | None:
        """The least recently used evictable entry, exactly. O(pinned)."""
        for key in self._order:
            entry = self._entries.get(key)
            if entry is not None and not entry.pinned and key != exclude:
                return key
        return None

    def _push(self, key: ExpertKey) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (self.value(key), self._seq, key))

    def _choose_victim(self, exclude: ExpertKey | None = None) -> ExpertKey | None:
        """The lowest-value unpinned entry, approximately.

        Approximately, and deliberately: an entry's value moves as the
        tracker's window slides, so a score stored at push time is a
        lower bound that ages. Popping discards entries that are gone or
        pinned, and re-pushes any whose true value has risen more than
        ``revalidate_tolerance`` above the stored one. Within the tolerance
        the candidate is taken as the minimum.

        The budget bounds the work: a window that has drifted for everyone
        could otherwise make one eviction re-push the whole heap. Exceeding
        it falls back to the exact scan and counts the event, so the
        approximation cannot hide.

        Asking does not consume: the chosen key is pushed back before it is
        returned, so the heap stays a superset of the resident keys whether
        or not the caller goes on to evict. Removal is lazy — a dropped
        key's heap entry is skipped the next time it surfaces.
        """
        deferred = 0
        candidate: ExpertKey | None = None
        while self._heap:
            stored, seq, key = heapq.heappop(self._heap)
            entry = self._entries.get(key)
            if entry is None or entry.pinned or key == exclude:
                continue                      # gone, pinned, or the newcomer
            true_value = self.value(key)
            if true_value > stored * (1.0 + self.revalidate_tolerance):
                heapq.heappush(self._heap, (true_value, seq, key))
                deferred += 1
                if deferred >= self.revalidate_budget:
                    break        # too much drift to approximate; measure exactly
                continue
            heapq.heappush(self._heap, (true_value, seq, key))
            candidate = key
            break

        front = self._lru_front(exclude)
        if candidate is None:
            if front is None:
                return None if not self._evictable(exclude) else self._scan_fallback(exclude)
            return front if deferred < self.revalidate_budget else self._scan_fallback(exclude)
        if front is not None and front != candidate and self.value(front) < self.value(candidate):
            return front
        return candidate
    def _scan_fallback(self, exclude: ExpertKey | None) -> ExpertKey | None:
        """The exact scan, counted. A silent O(n) is what the heap exists to rule out."""
        if not self._evictable(exclude):
            return None
        self.heap_fallbacks += 1
        return self._scan_victim(exclude)

    def _evictable(self, exclude: ExpertKey | None) -> bool:
        return any(not e.pinned and k != exclude for k, e in self._entries.items())

    def _scan_victim(self, exclude: ExpertKey | None = None) -> ExpertKey | None:
        """The exact lowest-value entry. The fallback, and what tests compare against."""
        candidates = [k for k, e in self._entries.items() if not e.pinned and k != exclude]
        if not candidates:
            return None
        return min(candidates, key=lambda k: (self.value(k), self._entries[k].last_hit))

    def _drop(self, key: ExpertKey, *, evicted: bool) -> None:
        entry = self._entries.pop(key)
        self._order.pop(key, None)
        self._bytes -= entry.nbytes
        if evicted:
            self.stats.evictions += 1
            self.stats.bytes_evicted += entry.nbytes

    # -- planning -------------------------------------------------------

    def missing(self, keys: Iterable[ExpertKey]) -> list[ExpertKey]:
        """Which of these are not resident. Does not count as a lookup."""
        return [k for k in keys if k not in self._entries]

    def would_evict_for(self, nbytes: int) -> list[ExpertKey]:
        """Which entries admitting ``nbytes`` would cost, without admitting it.

        Lets a caller decide whether a prefetch is worth its collateral before
        it happens, rather than discovering it afterwards.
        """
        if nbytes <= self.free_bytes:
            return []
        freed = self.free_bytes
        victims: list[ExpertKey] = []
        blocked = set()
        while freed < nbytes:
            candidates = [k for k, e in self._entries.items()
                          if not e.pinned and k not in blocked]
            if not candidates:
                break
            victim = min(candidates, key=lambda k: (self.value(k), self._entries[k].last_hit))
            victims.append(victim)
            blocked.add(victim)
            freed += self._entries[victim].nbytes
        return victims
