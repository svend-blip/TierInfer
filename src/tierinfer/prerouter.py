"""A trainable prerouter: predict a layer's experts from the layer before.

The heuristic predictors in `tierinfer.predict` count. This one learns. For
every MoE layer L it holds a linear map from the multi-hot routing of layer
L−1 (what this token just used) to a score per expert of L, trained by
gradient descent on a multi-label logistic loss over captured traces. It is
the smallest model that can express what `Transition` counts and more —
weights instead of co-occurrence counts, negative evidence as well as
positive — and it answers the scope's requirement for a trainable prerouter
in the only honest way: by being measured against the counting predictors
on the same traces, and kept only where it wins.

It implements the `Predictor` interface, so it drops into the prefetcher and
the loader unchanged; `observe` keeps learning online (one SGD step per
token) so a model shipped from one prompt class adapts to another.

Persistence is a `.npz` (weights, biases, layer ids, expert count, a schema
version); `load` refuses a file for another model shape.

numpy is optional for TierInfer and required here. Without it the class
says so on construction rather than run a Python loop over 160×160 matrices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from .predict import Frequency, Predictor, Routing

try:
    import numpy as np
except ImportError:                          # pragma: no cover — depends on the host
    np = None

SCHEMA = 1


class PrerouterUnavailable(RuntimeError):
    pass


class Prerouter(Predictor):
    """Linear multi-label model per layer: P(expert e at L | routing at L−1)."""

    name = "prerouter"

    def __init__(self, layers: Sequence[int], n_expert: int, *, lr: float = 0.05,
                 l2: float = 1e-4, online: bool = True, seed: int = 0) -> None:
        if np is None:
            raise PrerouterUnavailable("the prerouter needs numpy; install it or use the counting predictors")
        self.layers = sorted(int(l) for l in layers)
        self.n = int(n_expert)
        self.lr = lr
        self.l2 = l2
        self.online = online
        rng = np.random.default_rng(seed)
        # W[L]: (n_prev_experts, n_experts), b[L]: (n_experts,) for each layer with a predecessor
        self.W = {l: rng.normal(0, 0.01, (self.n, self.n)).astype(np.float32) for l in self.layers[1:]}
        self.b = {l: np.zeros(self.n, dtype=np.float32) for l in self.layers[1:]}
        self.fallback = Frequency()          # the first layer has no predecessor
        self.steps = 0
        self.trained_tokens = 0

    # -- the model ------------------------------------------------------------

    def _x(self, experts) -> "np.ndarray":
        x = np.zeros(self.n, dtype=np.float32)
        x[[int(e) for e in experts if 0 <= int(e) < self.n]] = 1.0
        return x

    def _logits(self, layer: int, prev: Sequence[int]) -> "np.ndarray":
        return self._x(prev) @ self.W[layer] + self.b[layer]

    def _step(self, layer: int, prev: Sequence[int], actual: Sequence[int]) -> float:
        x = self._x(prev)
        z = x @ self.W[layer] + self.b[layer]
        p = 1.0 / (1.0 + np.exp(-z))
        y = self._x(actual)
        g = p - y                                   # dLoss/dz for logistic loss
        self.W[layer] -= self.lr * (np.outer(x, g) + self.l2 * self.W[layer])
        self.b[layer] -= self.lr * g
        eps = 1e-7
        return float(-(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean())

    # -- Predictor ------------------------------------------------------------

    def observe(self, routing: Routing) -> None:
        self.fallback.observe(routing)
        if not self.online:
            return
        self._learn(routing)

    def _learn(self, routing: Routing) -> float:
        present = sorted(l for l in routing if l in self.W or l == self.layers[0])
        loss = 0.0
        n = 0
        for prev, cur in zip(present, present[1:]):
            if cur in self.W:
                loss += self._step(cur, routing[prev], routing[cur])
                n += 1
        self.steps += n
        self.trained_tokens += 1
        return loss / n if n else 0.0

    def score(self, layer: int, sofar: Routing) -> dict[int, float]:
        prev_layers = [l for l in sofar if l < layer]
        if layer not in self.W or not prev_layers:
            return self.fallback.score(layer, sofar)
        z = self._logits(layer, sofar[max(prev_layers)])
        p = 1.0 / (1.0 + np.exp(-z))
        return {int(e): float(p[e]) for e in range(self.n)}

    # -- training and persistence -------------------------------------------

    def train(self, trace: Sequence[Routing], *, epochs: int = 3) -> list[float]:
        """Offline: several passes over captured routings. Returns mean loss per epoch."""
        was = self.online
        self.online = False
        losses = []
        try:
            for _ in range(epochs):
                total = 0.0
                for routing in trace:
                    total += self._learn(routing)
                losses.append(total / max(1, len(trace)))
        finally:
            self.online = was
        return losses

    def save(self, path: str | Path) -> None:
        path = Path(path)
        meta = {"schema": SCHEMA, "layers": self.layers, "n_expert": self.n, "lr": self.lr,
                "l2": self.l2, "steps": self.steps, "trained_tokens": self.trained_tokens}
        arrays = {f"W{l}": self.W[l] for l in self.W} | {f"b{l}": self.b[l] for l in self.b}
        np.savez(path, meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8), **arrays)

    @classmethod
    def load(cls, path: str | Path, *, expect_layers: Sequence[int] | None = None,
             expect_experts: int | None = None) -> "Prerouter":
        if np is None:
            raise PrerouterUnavailable("the prerouter needs numpy")
        with np.load(Path(path)) as z:
            meta = json.loads(bytes(z["meta"]).decode())
            if meta.get("schema") != SCHEMA:
                raise ValueError(f"prerouter file schema {meta.get('schema')}, this reader knows {SCHEMA}")
            if expect_experts is not None and meta["n_expert"] != expect_experts:
                raise ValueError(f"prerouter was trained for {meta['n_expert']} experts, model has {expect_experts}")
            if expect_layers is not None and sorted(meta["layers"]) != sorted(int(l) for l in expect_layers):
                raise ValueError("prerouter was trained for a different set of MoE layers")
            pr = cls(meta["layers"], meta["n_expert"], lr=meta["lr"], l2=meta["l2"])
            for l in pr.W:
                pr.W[l] = z[f"W{l}"].astype(np.float32)
                pr.b[l] = z[f"b{l}"].astype(np.float32)
            pr.steps = meta["steps"]
            pr.trained_tokens = meta["trained_tokens"]
        return pr
