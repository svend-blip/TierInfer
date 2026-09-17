"""What fits on the card, and what it costs to put it there.

VRAM is the tier where the budget is hardest and the arithmetic is least
forgiving. A 32 GB card holding a 56.46 GB model is not a question of
policy until the other claims on that memory have been settled, and they
are large: at this model's full 131 072-token context the KV cache alone is
about 24 GB, which is most of the card before a single weight is placed.

So the budget comes first and is derived, not guessed:

    weights = total − reserve − kv_cache − runtime_overhead

The KV term is computed from the model's own metadata and the context the
caller actually intends to use, and it is **verified against llama.cpp's own
allocation** at three context lengths on this model:

    context   this formula   llama.cpp reports
       2 048       376 MiB             376 MiB
       8 192     1 504 MiB           1 504 MiB
      16 384     3 008 MiB           3 008 MiB

``runtime_overhead`` is the compute buffer and scratch a forward pass needs.
It is *not* derived, because it depends on the runtime rather than the model
— so it is measured instead: llama.cpp reports 330, 328 and 320 MiB at those
same three contexts, near enough constant, which also confirms it scales with
the batch rather than the context. The default here is 512 MB, above every
measurement, and it is labelled a measurement of one runtime rather than a
property of the model.

The device is reached through ``ctypes`` against ``libcudart``, the same way
``mincore`` and ``posix_fadvise`` are reached elsewhere in this project.
TierInfer does not depend on torch or cupy to move bytes to a card, and a
host without a CUDA runtime gets a clear error rather than an import failure
at module load.

**Residency here is explicit and bounded.** Nothing is placed on the device
without a budget saying there is room, and the manager refuses rather than
letting the allocator decide — an out-of-memory in the middle of a token is
not a policy outcome, it is a crash.
"""

from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass, field
from typing import Iterable

GB = 1024 ** 3
MB = 1024 ** 2

# cudaMemcpyKind
_H2D = 1
_D2H = 2


class CudaError(RuntimeError):
    pass


class CudaUnavailable(CudaError):
    """No CUDA runtime on this host. Not a failure — a fact to report."""


# -- the runtime --------------------------------------------------------


_CANDIDATES = ("libcudart.so", "libcudart.so.13", "libcudart.so.12",
               "/usr/local/cuda/lib64/libcudart.so")


class CudaRuntime:
    """Enough of the CUDA runtime to budget, allocate and copy.

    Deliberately small. Every call checks its status and raises with the
    runtime's own error string, because a silently ignored CUDA error shows
    up later as wrong numbers rather than as a failure.
    """

    def __init__(self, library: str | None = None) -> None:
        names = (library,) if library else _CANDIDATES
        self.lib = None
        for name in names:
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if self.lib is None:
            raise CudaUnavailable(f"no CUDA runtime found (tried {', '.join(names)})")
        self._bind()

    def _bind(self) -> None:
        l = self.lib
        sz = ctypes.c_size_t
        l.cudaMemGetInfo.argtypes = [ctypes.POINTER(sz), ctypes.POINTER(sz)]
        l.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        l.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), sz]
        l.cudaFree.argtypes = [ctypes.c_void_p]
        l.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), sz, ctypes.c_uint]
        l.cudaFreeHost.argtypes = [ctypes.c_void_p]
        l.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, sz, ctypes.c_int]
        l.cudaDeviceSynchronize.argtypes = []
        l.cudaGetErrorString.argtypes = [ctypes.c_int]
        l.cudaGetErrorString.restype = ctypes.c_char_p

    def _check(self, rc: int, what: str) -> None:
        if rc != 0:
            msg = self.lib.cudaGetErrorString(rc)
            raise CudaError(f"{what}: {msg.decode() if msg else f'cuda error {rc}'}")

    # -- information ----------------------------------------------------

    def device_count(self) -> int:
        n = ctypes.c_int()
        self._check(self.lib.cudaGetDeviceCount(ctypes.byref(n)), "cudaGetDeviceCount")
        return n.value

    def memory_info(self) -> "DeviceMemory":
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        self._check(self.lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)),
                    "cudaMemGetInfo")
        return DeviceMemory(free=free.value, total=total.value)

    # -- memory ---------------------------------------------------------

    def malloc(self, nbytes: int) -> int:
        p = ctypes.c_void_p()
        self._check(self.lib.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(nbytes)),
                    f"cudaMalloc({nbytes})")
        return p.value

    def free(self, ptr: int) -> None:
        self._check(self.lib.cudaFree(ctypes.c_void_p(ptr)), "cudaFree")

    def host_alloc(self, nbytes: int) -> int:
        """Page-locked host memory. Pinned transfers are the ones worth timing."""
        p = ctypes.c_void_p()
        self._check(self.lib.cudaHostAlloc(ctypes.byref(p), ctypes.c_size_t(nbytes), 0),
                    f"cudaHostAlloc({nbytes})")
        return p.value

    def host_free(self, ptr: int) -> None:
        self._check(self.lib.cudaFreeHost(ctypes.c_void_p(ptr)), "cudaFreeHost")

    def copy_to_device(self, dst: int, src: int, nbytes: int) -> float:
        """Host to device, synchronously, returning seconds actually taken."""
        self.synchronize()
        t0 = time.perf_counter()
        self._check(self.lib.cudaMemcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src),
                                        ctypes.c_size_t(nbytes), _H2D), "cudaMemcpy H2D")
        self.synchronize()
        return time.perf_counter() - t0

    def copy_to_host(self, dst: int, src: int, nbytes: int) -> float:
        self.synchronize()
        t0 = time.perf_counter()
        self._check(self.lib.cudaMemcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src),
                                        ctypes.c_size_t(nbytes), _D2H), "cudaMemcpy D2H")
        self.synchronize()
        return time.perf_counter() - t0

    def synchronize(self) -> None:
        self._check(self.lib.cudaDeviceSynchronize(), "cudaDeviceSynchronize")


@dataclass(frozen=True)
class DeviceMemory:
    free: int
    total: int

    @property
    def used(self) -> int:
        return self.total - self.free


# -- the budget ---------------------------------------------------------


@dataclass(frozen=True)
class VramBudget:
    """What each claim on the card takes, and what is left for weights.

    Every figure is bytes. ``weights`` is what remains and may be negative,
    which is a real answer: it says this context does not fit on this card
    whatever the residency policy does.
    """

    total: int
    reserve: int
    kv_cache: int
    runtime_overhead: int
    context_length: int
    kv_bits: int
    layers: int

    @property
    def weights(self) -> int:
        return self.total - self.reserve - self.kv_cache - self.runtime_overhead

    @property
    def fits(self) -> bool:
        return self.weights > 0

    def experts(self, expert_bytes: int, floor_bytes: int = 0) -> int:
        """How many experts fit alongside a resident floor. Never negative."""
        room = self.weights - floor_bytes
        return max(0, room // expert_bytes) if expert_bytes > 0 else 0

    @classmethod
    def kv_bytes_per_token(cls, metadata: dict, *, layers: int, kv_bits: int = 16) -> int:
        """KV cache for one token across all attention layers.

        Read from the model's own metadata: heads_kv × (key_len + value_len)
        × 2 tensors' worth of elements, per layer. Grouped-query attention is
        why this is small enough to be worth computing rather than assuming —
        this model has 96 query heads against 8 key/value heads.
        """
        arch = metadata.get("general.architecture", "")
        def m(key, default=None):
            v = metadata.get(f"{arch}.{key}", default)
            if v is None:
                raise CudaError(f"model metadata has no {arch}.{key}; "
                                "the KV budget cannot be derived for it")
            return int(v)
        n_kv = m("attention.head_count_kv")
        k_len = m("attention.key_length")
        v_len = m("attention.value_length")
        return layers * n_kv * (k_len + v_len) * kv_bits // 8

    #: Measured from llama.cpp b9888 on this model: 330, 328 and 320 MiB at
    #: contexts 2 048, 8 192 and 16 384 — constant in context, so it belongs
    #: to the runtime and the batch, not to the context. The default sits
    #: above all three.
    MEASURED_RUNTIME_OVERHEAD = 512 * MB

    @classmethod
    def from_model(cls, metadata: dict, *, total_bytes: int, context_length: int,
                   layers: int, kv_bits: int = 16,
                   reserve_bytes: int = 512 * MB,
                   runtime_overhead: int | None = None) -> "VramBudget":
        """Derive the budget for a context the caller actually intends to use.

        ``reserve`` is the driver's own footprint and whatever else shares the
        card. It should be *measured* — ``CudaRuntime.memory_info().used``
        before anything of ours is placed — rather than assumed: on this host
        an unrelated service holds half a gigabyte and a CUDA context another
        gigabyte on top.
        """
        kv = cls.kv_bytes_per_token(metadata, layers=layers, kv_bits=kv_bits) * context_length
        return cls(total=total_bytes, reserve=reserve_bytes, kv_cache=kv,
                   runtime_overhead=(cls.MEASURED_RUNTIME_OVERHEAD
                                     if runtime_overhead is None else runtime_overhead),
                   context_length=context_length, kv_bits=kv_bits, layers=layers)

    def explain(self) -> str:
        rows = [("total", self.total), ("reserve (driver, context, others)", -self.reserve),
                (f"KV cache ({self.context_length} tokens, {self.kv_bits}-bit)", -self.kv_cache),
                ("runtime overhead (compute buffers)", -self.runtime_overhead),
                ("left for weights", self.weights)]
        return "\n".join(f"{name:<44}{v / GB:>9.2f} GB" for name, v in rows)


# -- residency ----------------------------------------------------------


@dataclass
class VramStats:
    hits: int = 0
    misses: int = 0
    transfers: int = 0
    evictions: int = 0
    bytes_transferred: int = 0
    transfer_seconds: float = 0.0
    refused: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    @property
    def bytes_per_second(self) -> float:
        return (self.bytes_transferred / self.transfer_seconds
                if self.transfer_seconds > 0 else 0.0)

    @property
    def mean_transfer_seconds(self) -> float:
        return self.transfer_seconds / self.transfers if self.transfers else 0.0


class VramResidency:
    """Which experts live on the device, inside a budget that was computed.

    Slots are allocated once, up front, and reused. Allocating per expert
    would put ``cudaMalloc`` in the token loop and fragment the heap; a fixed
    pool makes residency a bookkeeping question and the device footprint a
    constant that can be checked against the budget rather than hoped about.

    The eviction policy is ``ExpertCache``, which on measured routing is
    exactly LRU (see `benchmarks/REAL-ROUTING.md` — the frequency-led policy
    it started as lost to LRU by 6 to 9 points, and the reason is in that
    file). Reusing it means this tier and the RAM tier cannot drift apart in
    what they believe about an expert's worth.

    Nothing is admitted without a slot. When the pool is full an eviction is
    forced; when a caller asks for more than the pool holds, ``admit``
    refuses and says so, because an out-of-memory in the middle of a token is
    a crash rather than a policy.
    """

    def __init__(self, runtime: CudaRuntime, slot_bytes: int, slots: int,
                 tracker=None) -> None:
        if slot_bytes <= 0 or slots <= 0:
            raise ValueError("residency needs at least one slot of at least one byte")
        from .cache import ExpertCache        # local: cache does not import vram
        self.runtime = runtime
        self.slot_bytes = slot_bytes
        self.slots = slots
        self.stats = VramStats()
        self._pointers: list[int] = []
        self._assigned: dict[object, int] = {}
        self._free: list[int] = []
        self._policy = ExpertCache(slots * slot_bytes, tracker)
        for _ in range(slots):
            self._pointers.append(runtime.malloc(slot_bytes))
        self._free = list(self._pointers)

    @classmethod
    def from_budget(cls, runtime: CudaRuntime, budget: VramBudget, *,
                    expert_bytes: int, floor_bytes: int = 0, tracker=None
                    ) -> "VramResidency":
        """Size the pool to what the budget actually leaves for experts."""
        n = budget.experts(expert_bytes, floor_bytes)
        if n <= 0:
            raise CudaError(
                f"the budget leaves {(budget.weights - floor_bytes) / GB:.2f} GB for "
                f"experts of {expert_bytes / MB:.2f} MB — this context does not fit "
                "on this card alongside the resident floor")
        return cls(runtime, expert_bytes, n, tracker=tracker)

    # -- use ------------------------------------------------------------

    @property
    def device_bytes(self) -> int:
        return self.slot_bytes * self.slots

    def __contains__(self, key: object) -> bool:
        return key in self._assigned

    def __len__(self) -> int:
        return len(self._assigned)

    def lookup(self, key: object) -> int | None:
        """The device pointer for a resident expert, counting the lookup."""
        ptr = self._assigned.get(key)
        if ptr is None:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        self._policy.get(key)
        return ptr

    def admit(self, key: object, host_ptr: int, nbytes: int) -> int:
        """Place an expert on the device, evicting to make room. Returns its pointer.

        ``host_ptr`` should address page-locked memory: pinned transfers
        measured 27.6 GB/s here against 16.0 for pageable, which on a
        9.97 MB expert is 0.36 ms against 0.62.
        """
        if nbytes > self.slot_bytes:
            self.stats.refused += 1
            raise CudaError(f"{key!r} needs {nbytes} bytes, slots hold {self.slot_bytes}")
        existing = self._assigned.get(key)
        if existing is not None:
            return existing
        if not self._free:
            self._evict_one(exclude=key)
        ptr = self._free.pop()
        self.stats.transfer_seconds += self.runtime.copy_to_device(ptr, host_ptr, nbytes)
        self.stats.transfers += 1
        self.stats.bytes_transferred += nbytes
        self._assigned[key] = ptr
        self._policy.put(key, ptr, self.slot_bytes)
        return ptr

    def _evict_one(self, exclude: object | None = None) -> None:
        victim = self._policy._choose_victim(exclude=exclude)
        if victim is None or victim not in self._assigned:
            # The policy and the assignment disagreed. Rather than guess,
            # take the oldest assignment: a stuck pool is worse than an
            # imperfect victim.
            victim = next(iter(self._assigned), None)
            if victim is None:
                raise CudaError("no slot is free and nothing is resident to evict")
        self.release(victim)

    def release(self, key: object) -> None:
        """Give a slot back without freeing device memory."""
        ptr = self._assigned.pop(key, None)
        if ptr is None:
            return
        self._free.append(ptr)
        self.stats.evictions += 1
        if key in self._policy:
            self._policy._drop(key, evicted=True)

    def close(self) -> None:
        for ptr in self._pointers:
            self.runtime.free(ptr)
        self._pointers.clear()
        self._assigned.clear()
        self._free.clear()

    def __enter__(self) -> "VramResidency":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
