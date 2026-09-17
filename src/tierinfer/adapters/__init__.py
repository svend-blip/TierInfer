"""Adapters to the runtimes TierInfer has to live beside.

Goal 6 measured what an adapter can and cannot be. Residency in another
process's mmap cannot be shaped from outside it — `posix_fadvise` succeeds and
does nothing — so an adapter either owns the loading path or it does not touch
residency at all. These do not: each one supplies a **configuration** derived
from the host and the model, and takes **telemetry** back in a single schema.

That is a smaller claim than it sounds. A runtime that already tiers experts
well, as FreeToken does, does not need TierInfer's policy; it needs the
budget computed correctly and its own numbers put in a comparable shape. A
runtime that does neither is where `tierinfer.storage` belongs instead.
"""

from __future__ import annotations
