"""A TierInfer-served buffer, from Python — the pair of hands FreeToken needs.

llama.cpp gets its served mappings from the preloaded shim, which turns a
file mapping into a userfaultfd region behind its back. A Python runtime
that owns its buffers (FreeToken's ``HostBank``) can ask for one directly:

    region = TieredRegion(sock, tag="gate_up_packed#L00007", nbytes=n, logical_off=off)
    t = torch.frombuffer(region.buffer, dtype=torch.uint8)   # zero pages until touched
    ...
    region.route(layer, expert_ids)                           # what the router chose
    region.close()

The buffer is an anonymous, private, no-reserve mapping registered for
MISSING faults; the descriptor, the base and the buffer's identity in the
model's logical byte region go to ``tierinfer serve`` as
``MAP <tag> <base> <len> <logical_off>``. From then on the first touch of
any page is a fault the server answers with the whole expert row from the
FTW shard; evictions arrive on a second socket as ``EVICT <addr> <len>``
and are applied here with ``MADV_DONTNEED`` — refused when they fall
outside the buffer. ``close()`` tells the server (``UNMAP``) before the
memory goes, so nothing is ever evicted into an address that is no longer
ours.

Only the process that owns the buffer can register it; the server never
touches this process except through the descriptor it was handed.
"""

from __future__ import annotations

import array
import ctypes
import ctypes.util
import errno
import fcntl
import mmap
import os
import socket
import struct
import threading

# userfaultfd — x86-64 numbers; the ioctl numbers are architecture-independent
_SYS_USERFAULTFD = 323
_UFFD_USER_MODE_ONLY = 1
_UFFD_API = 0xAA
_UFFDIO_API = 0xC018AA3F
_UFFDIO_REGISTER = 0xC020AA00
_UFFDIO_REGISTER_MODE_MISSING = 1
PAGE = mmap.PAGESIZE

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_libc.syscall.restype = ctypes.c_long


class ClientError(RuntimeError):
    pass


def _errno_text() -> str:
    return os.strerror(ctypes.get_errno())


def open_userfaultfd() -> int:
    """One descriptor per process is enough; regions are registered on it."""
    fd = _libc.syscall(_SYS_USERFAULTFD, os.O_CLOEXEC | os.O_NONBLOCK | _UFFD_USER_MODE_ONLY)
    if fd < 0:
        raise ClientError(f"userfaultfd: {_errno_text()} (needs vm.unprivileged_userfaultfd=1 or CAP_SYS_PTRACE)")
    api = bytearray(struct.pack("<QQQ", _UFFD_API, 0, 0))
    try:
        fcntl.ioctl(fd, _UFFDIO_API, api)
    except OSError as e:
        os.close(fd)
        raise ClientError(f"UFFDIO_API: {e}") from e
    return fd


class _Session:
    """The two sockets to one server, shared by every region of this process."""

    def __init__(self, sock_path: str) -> None:
        self.path = sock_path
        self.pid = os.getpid()
        self.ctl = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.ctl.connect(sock_path)
        self.ctl.sendall(f"HELLO ctl {self.pid}\n".encode())
        self.evict = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.evict.connect(sock_path)
        self.evict.sendall(f"HELLO evict {self.pid}\n".encode())
        self.uffd = open_userfaultfd()
        self.lock = threading.Lock()
        self.regions: dict[int, "TieredRegion"] = {}          # base -> region
        self.evictions = 0
        self.refused = 0
        self._ctl_file = self.ctl.makefile("rb", buffering=0)
        self._thread = threading.Thread(target=self._evict_loop, daemon=True, name="tierinfer-evict")
        self._thread.start()

    def announce(self, region: "TieredRegion") -> None:
        msg = f"MAP {region.tag} {region.base:x} {region.nbytes} {region.logical_off}\n".encode()
        with self.lock:
            self.ctl.sendmsg([msg], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [self.uffd]))])
            reply = self._ctl_file.readline().decode().strip()
        if not reply.startswith("OK"):
            raise ClientError(f"server did not accept {region.tag}: {reply or 'closed'}")
        with self.lock:
            self.regions[region.base] = region

    def say(self, line: str) -> None:
        with self.lock:
            self.ctl.sendall((line + "\n").encode())

    def route(self, layer: int, expert_ids, *, after: bool = False) -> None:
        """What the router chose for this layer, this step. ``expert_ids``: one
        row of ids (a token) or rows (a batch); negative ids are skipped (a
        hybrid split marks the other side's routes -1).

        ``after=True`` says the step has already run — the runtime could only
        read the ids back afterwards (CUDA-graph replay leaves Python out of
        the step). The server then learns the routing and the token boundary
        but scores hits by whether the expert had to be *faulted in* during
        the step, not by whether it is resident now (it always is, by then).
        """
        rows = expert_ids
        if hasattr(rows, "tolist"):
            rows = rows.tolist()
        if rows and not isinstance(rows[0], (list, tuple)):
            rows = [rows]
        rows = [[int(e) for e in row if int(e) >= 0] for row in rows]
        rows = [r for r in rows if r]
        if not rows:
            return
        body = ";".join(",".join(str(e) for e in row) for row in rows)
        verb = "ROUTED" if after else "ROUTE"
        self.say(f"{verb} {layer} {len(rows)} {len(rows[0])} {body}")

    def _owner(self, addr: int, n: int) -> "TieredRegion | None":
        for base, r in self.regions.items():
            if base <= addr and addr + n <= base + r.alen and not r.closed:
                return r
        return None

    def _evict_loop(self) -> None:
        f = self.evict.makefile("rb", buffering=0)
        for raw in f:
            line = raw.decode().strip()
            if line.startswith("EVICT "):
                _, a, n = line.split()
                addr, n = int(a, 16), int(n)
                with self.lock:
                    r = self._owner(addr, n)
                if r is None:
                    self.refused += 1        # not ours, or no longer ours: never DONTNEED it
                    continue
                if _libc.madvise(ctypes.c_void_p(addr), n, mmap.MADV_DONTNEED) == 0:
                    self.evictions += 1
            elif line == "PING":
                self.evict.sendall(b"PONG\n")


_sessions: dict[str, _Session] = {}
_sessions_lock = threading.Lock()


def session(sock_path: str) -> _Session:
    with _sessions_lock:
        s = _sessions.get(sock_path)
        if s is None or s.pid != os.getpid():
            s = _sessions[sock_path] = _Session(sock_path)
        return s


class TieredRegion:
    """One buffer standing in for ``nbytes`` of the model's logical region at
    ``logical_off``, materialised per expert by the server on first touch."""

    def __init__(self, sock_path: str, *, tag: str, nbytes: int, logical_off: int) -> None:
        if nbytes <= 0:
            raise ClientError("a region needs a positive size")
        if " " in tag:
            raise ClientError("the tag names the buffer in a text protocol: no spaces")
        self.tag = tag
        self.nbytes = nbytes
        self.logical_off = logical_off
        self.alen = (nbytes + PAGE - 1) & ~(PAGE - 1)
        self.closed = False
        self.session = session(sock_path)
        base = _libc.mmap(None, self.alen, mmap.PROT_READ | mmap.PROT_WRITE,
                          mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | 0x4000, -1, 0)   # 0x4000 = MAP_NORESERVE
        if base in (None, ctypes.c_void_p(-1).value):
            raise ClientError(f"anonymous mmap of {self.alen} bytes: {_errno_text()}")
        self.base = base
        _libc.madvise(ctypes.c_void_p(base), self.alen, 15)                       # MADV_NOHUGEPAGE
        reg = bytearray(struct.pack("<QQQQ", base, self.alen, _UFFDIO_REGISTER_MODE_MISSING, 0))
        try:
            fcntl.ioctl(self.session.uffd, _UFFDIO_REGISTER, reg)
        except OSError as e:
            _libc.munmap(ctypes.c_void_p(base), self.alen)
            raise ClientError(f"UFFDIO_REGISTER {self.alen} bytes: {e}") from e
        try:
            self.session.announce(self)
        except Exception:
            _libc.munmap(ctypes.c_void_p(base), self.alen)
            raise
        self.buffer = (ctypes.c_uint8 * nbytes).from_address(base)

    @property
    def addr(self) -> int:
        return self.base

    def memoryview(self) -> memoryview:
        return memoryview(self.buffer).cast("B")

    def route(self, layer: int, expert_ids, *, after: bool = False) -> None:
        """See ``_Session.route``."""
        self.session.route(layer, expert_ids, after=after)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        with self.session.lock:
            self.session.regions.pop(self.base, None)
        self.session.say(f"UNMAP {self.base:x} {self.alen}")        # the server forgets first
        _libc.munmap(ctypes.c_void_p(self.base), self.alen)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
