"""Guessing which experts a token will need, and being honest about it.

Prefetching needs a guess made before routing happens. Routing is the
authority and stays the authority: a prediction may decide what to *have
ready*, never what to *use*. Everything here produces rankings and
confidences; nothing here is allowed to answer the question "which expert
does this token use".

Four predictors, each a different claim about where the signal is:

``Frequency``   the experts this layer uses most often, ignoring context.
                The floor. Anything that cannot beat it is not earning the
                memory it costs.
``Persistence`` the experts this layer used for the previous token. MoE
                routing is strongly autocorrelated within a prompt, and this
                exploits only that.
``Transition``  a first-order model over the previous layer's choices:
                P(expert e at layer L | the experts routed at layer L-1).
                The only one that uses the current token's own routing so
                far, which is available because layers run in order.
``Blend``       a weighted sum of the three, weights settable and reported.

Evaluation is ``recall@k``: of the experts a token actually routed to, how
many were in the top k of the prediction. That is the quantity prefetching
cares about — a predicted expert that goes unused costs bandwidth, but a used
expert that went unpredicted costs a stall, and stalls are the expensive
kind.

**What the numbers here do and do not prove.** Evaluated against a synthetic
trace, a predictor is being scored on a generator someone wrote, and the
generator's assumptions are the predictor's advantage. ``evaluate`` takes any
iterable of per-token routings, so the moment real traces exist from a
running model, the same harness scores them without changing. Until then,
treat a synthetic result as a statement about the code, not about the model.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

ExpertId = int
LayerId = int
Routing = Mapping[LayerId, Sequence[ExpertId]]


@dataclass(frozen=True)
class Prediction:
    """Ranked experts for one layer, with a confidence in 0..1 each."""

    layer: LayerId
    experts: tuple[ExpertId, ...]
    confidence: tuple[float, ...]

    def top(self, k: int) -> tuple[ExpertId, ...]:
        return self.experts[:k]

    def as_scores(self) -> dict[ExpertId, float]:
        return dict(zip(self.experts, self.confidence))


class Predictor:
    """Common shape. Subclasses fill in ``score``."""

    name = "predictor"

    def observe(self, routing: Routing) -> None:
        """Learn from one token's actual routing, after the fact."""
        raise NotImplementedError

    def score(self, layer: LayerId, sofar: Routing) -> dict[ExpertId, float]:
        """Unnormalised scores for this layer, given the routing so far."""
        raise NotImplementedError

    def predict(self, layer: LayerId, sofar: Routing | None = None) -> Prediction:
        scores = self.score(layer, sofar or {})
        if not scores:
            return Prediction(layer, (), ())
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        total = sum(v for _, v in ranked) or 1.0
        return Prediction(layer,
                          tuple(e for e, _ in ranked),
                          tuple(v / total for _, v in ranked))


class Frequency(Predictor):
    """How often each expert has been routed to in this layer. The floor."""

    name = "frequency"

    def __init__(self) -> None:
        self.counts: dict[LayerId, dict[ExpertId, int]] = defaultdict(lambda: defaultdict(int))

    def observe(self, routing: Routing) -> None:
        for layer, experts in routing.items():
            for e in experts:
                self.counts[layer][e] += 1

    def score(self, layer: LayerId, sofar: Routing) -> dict[ExpertId, float]:
        return {e: float(c) for e, c in self.counts.get(layer, {}).items()}


class Persistence(Predictor):
    """Whatever this layer used last token, weighted above everything else.

    Falls back to frequency where there is no previous token, so it is never
    empty — an empty prediction is indistinguishable from a confident wrong
    one at the call site, and only one of those is recoverable.
    """

    name = "persistence"

    def __init__(self, decay: float = 0.6, depth: int = 3) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be strictly between 0 and 1")
        self.decay = decay
        self.depth = depth
        self.history: dict[LayerId, list[tuple[ExpertId, ...]]] = defaultdict(list)
        self.fallback = Frequency()

    def observe(self, routing: Routing) -> None:
        self.fallback.observe(routing)
        for layer, experts in routing.items():
            h = self.history[layer]
            h.append(tuple(experts))
            if len(h) > self.depth:
                h.pop(0)

    def score(self, layer: LayerId, sofar: Routing) -> dict[ExpertId, float]:
        h = self.history.get(layer, [])
        if not h:
            return self.fallback.score(layer, sofar)
        scores: dict[ExpertId, float] = defaultdict(float)
        weight = 1.0
        for experts in reversed(h):          # most recent token first
            for e in experts:
                scores[e] += weight
            weight *= self.decay
        return dict(scores)


class Transition(Predictor):
    """P(expert at this layer | the experts the previous layer just routed).

    Layers run in order, so by the time layer L is being prepared, layer L-1's
    routing for *this* token is known. That makes this the only predictor here
    with access to the current token rather than the previous one.
    """

    name = "transition"

    def __init__(self) -> None:
        # counts[layer][prev_expert][expert]
        self.counts: dict[LayerId, dict[ExpertId, dict[ExpertId, int]]] = \
            defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.fallback = Frequency()

    def observe(self, routing: Routing) -> None:
        self.fallback.observe(routing)
        layers = sorted(routing)
        for prev, cur in zip(layers, layers[1:]):
            for p in routing[prev]:
                for c in routing[cur]:
                    self.counts[cur][p][c] += 1

    def score(self, layer: LayerId, sofar: Routing) -> dict[ExpertId, float]:
        prev_layers = [l for l in sofar if l < layer]
        table = self.counts.get(layer)
        if not prev_layers or not table:
            return self.fallback.score(layer, sofar)
        prev = max(prev_layers)
        scores: dict[ExpertId, float] = defaultdict(float)
        for p in sofar[prev]:
            for c, n in table.get(p, {}).items():
                scores[c] += float(n)
        return dict(scores) or self.fallback.score(layer, sofar)


class Blend(Predictor):
    """A weighted sum of several predictors, each normalised to sum to 1.

    Normalising first matters: raw counts from Transition dwarf Persistence's
    small decayed weights, so an unnormalised sum is Transition wearing a
    blend's name.
    """

    name = "blend"

    def __init__(self, parts: Sequence[tuple[Predictor, float]]) -> None:
        if not parts:
            raise ValueError("a blend needs at least one part")
        if any(w < 0 for _, w in parts):
            raise ValueError("weights must not be negative")
        self.parts = list(parts)
        self.name = "blend(" + ", ".join(f"{p.name}:{w:g}" for p, w in parts) + ")"

    def observe(self, routing: Routing) -> None:
        seen: set[int] = set()
        for p, _ in self.parts:
            if id(p) not in seen:            # a shared sub-predictor learns once
                p.observe(routing)
                seen.add(id(p))

    def score(self, layer: LayerId, sofar: Routing) -> dict[ExpertId, float]:
        out: dict[ExpertId, float] = defaultdict(float)
        for p, w in self.parts:
            s = p.score(layer, sofar)
            total = sum(s.values())
            if total <= 0:
                continue
            for e, v in s.items():
                out[e] += w * v / total
        return dict(out)


class AdaptiveBlend(Predictor):
    """A blend whose weights are each part's own measured recall.

    Hand-set weights are a claim about which signal is strongest, and on a
    synthetic trace that claim gets tuned against the generator that produced
    it — which proves nothing about a real model. So no weights are set here.
    Each part carries an exponentially-weighted estimate of its own recall@k,
    updated from routings as they arrive, and contributes in proportion to it.

    That makes the blend track whichever signal is actually winning: the
    marginal where routing is dominated by popularity, context where it is
    not, and it moves between them without anyone choosing in advance. It is
    also the mechanism SCOPE goal 10 asks for, one level down.

    The estimate costs one top-k comparison per part per scored layer, so
    ``update_every`` samples rather than measuring every token.
    """

    name = "adaptive"

    def __init__(self, parts: Sequence[Predictor], *, k: int = 16,
                 alpha: float = 0.05, update_every: int = 4,
                 floor: float = 0.02, sharpness: float = 4.0) -> None:
        if not parts:
            raise ValueError("a blend needs at least one part")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if update_every < 1:
            raise ValueError("update_every must be at least 1")
        if sharpness < 1.0:
            raise ValueError("sharpness below 1 would flatten the measurement away")
        self.parts = list(parts)
        self.k = k
        self.alpha = alpha
        self.update_every = update_every
        self.floor = floor
        self.sharpness = sharpness
        # Start equal: no part is assumed better before anything is measured.
        self.recall = {id(p): 0.5 for p in self.parts}
        self._seen = 0

    def weights(self) -> dict[str, float]:
        """What each part currently counts for. Reported, never hidden.

        Weight is each part's recall relative to the best part, raised to
        ``sharpness``. Weighting by recall directly is too soft to act on
        what it measures: 0.55 against 0.46 is a ratio of 1.2, so the parts
        come out nearly equal however clearly one of them is winning. The
        ratio raised to the fourth separates them by about two to one, and
        collapses onto a single part when one dominates outright.

        The floor keeps a part that is losing now from being deleted, since
        which signal wins changes with the prompt.
        """
        best = max(self.recall.values()) or 1.0
        raw = {p.name: max((self.recall[id(p)] / best) ** self.sharpness, self.floor)
               for p in self.parts}
        total = sum(raw.values()) or 1.0
        return {n: v / total for n, v in raw.items()}

    def observe(self, routing: Routing) -> None:
        self._seen += 1
        if self._seen % self.update_every == 0:
            self._measure(routing)
        seen: set[int] = set()
        for p in self.parts:
            if id(p) not in seen:
                p.observe(routing)
                seen.add(id(p))

    def _measure(self, routing: Routing) -> None:
        """Score each part against this token, before it learns from it."""
        sofar: dict[LayerId, Sequence[ExpertId]] = {}
        hits = {id(p): 0 for p in self.parts}
        used = 0
        for layer in sorted(routing):
            actual = set(routing[layer])
            used += len(actual)
            for p in self.parts:
                scores = p.score(layer, sofar)
                top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:self.k]
                hits[id(p)] += len(actual & {e for e, _ in top})
            sofar[layer] = routing[layer]
        if not used:
            return
        for p in self.parts:
            r = hits[id(p)] / used
            self.recall[id(p)] = (1 - self.alpha) * self.recall[id(p)] + self.alpha * r

    def score(self, layer: LayerId, sofar: Routing) -> dict[ExpertId, float]:
        out: dict[ExpertId, float] = defaultdict(float)
        w = self.weights()
        for p in self.parts:
            s = p.score(layer, sofar)
            total = sum(s.values())
            if total <= 0:
                continue
            weight = w[p.name]
            for e, v in s.items():
                out[e] += weight * v / total
        return dict(out)


# -- evaluation ---------------------------------------------------------


@dataclass
class Score:
    predictor: str
    k: int
    tokens: int = 0
    layers: int = 0
    hits: int = 0
    used: int = 0
    predicted: int = 0
    per_layer_recall: dict[LayerId, tuple[int, int]] = field(default_factory=dict)

    @property
    def recall(self) -> float:
        """Of the experts actually routed to, the share that was in the top k."""
        return self.hits / self.used if self.used else 0.0

    @property
    def wasted(self) -> float:
        """Of the experts predicted, the share that went unused."""
        return 1.0 - (self.hits / self.predicted) if self.predicted else 0.0

    def worst_layers(self, n: int = 3) -> list[tuple[LayerId, float]]:
        rates = [(l, h / u) for l, (h, u) in self.per_layer_recall.items() if u]
        return sorted(rates, key=lambda kv: kv[1])[:n]


def evaluate(predictor: Predictor, trace: Iterable[Routing], k: int,
             *, warmup: int = 0) -> Score:
    """Score a predictor on a trace, predicting each token before observing it.

    The prediction for layer L may use this token's routing for layers below
    L, because those have already run — that is the real information a
    prefetcher has, and giving the predictor less would understate it as much
    as giving it the answer would overstate it.

    ``warmup`` tokens are observed without being scored, so a cold predictor's
    first guesses do not count against a model of steady-state behaviour.
    """
    score = Score(predictor=predictor.name, k=k)
    per_layer: dict[LayerId, list[int]] = defaultdict(lambda: [0, 0])
    for i, routing in enumerate(trace):
        if i < warmup:
            predictor.observe(routing)
            continue
        score.tokens += 1
        sofar: dict[LayerId, Sequence[ExpertId]] = {}
        for layer in sorted(routing):
            actual = set(routing[layer])
            top = set(predictor.predict(layer, sofar).top(k))
            hit = len(actual & top)
            score.layers += 1
            score.hits += hit
            score.used += len(actual)
            score.predicted += len(top)
            per_layer[layer][0] += hit
            per_layer[layer][1] += len(actual)
            sofar[layer] = routing[layer]
        predictor.observe(routing)
    score.per_layer_recall = {l: (h, u) for l, (h, u) in per_layer.items()}
    return score
