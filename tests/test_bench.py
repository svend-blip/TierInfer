"""Tests for the baseline measurement primitives.

None of these need the model file, and none of them need root. Where a
primitive cannot be exercised on a given host (no delegated memory
controller, a filesystem with no block device) the test says so and skips,
rather than asserting something the host cannot answer.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tierinfer.bench import (  # noqa: E402
    BenchError, DiskCounters, DiskDelta, device_for, disk_counters,
    drop_cache, measure, memory_controller_available, residency, run_limited,
    warm_cache, write_report,
)

MB = 1024 * 1024


@pytest.fixture
def datafile(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(os.urandom(8 * MB))
    return p


# -- disk counters ------------------------------------------------------


def test_counters_subtract_into_rates():
    a = DiskCounters("nvme0n1", reads=1000, sectors=200_000, ms=500, at=10.0)
    b = DiskCounters("nvme0n1", reads=1100, sectors=400_000, ms=700, at=12.0)
    d = b - a
    assert d.reads == 100
    assert d.bytes_read == 200_000 * 512
    assert d.seconds == pytest.approx(2.0)
    assert d.iops == pytest.approx(50.0)
    assert d.mean_read_bytes == pytest.approx(200_000 * 512 / 100)
    assert d.await_ms == pytest.approx(2.0)


def test_counters_from_different_devices_refuse_to_subtract():
    a = DiskCounters("nvme0n1", 1, 1, 1, 1.0)
    b = DiskCounters("sda", 2, 2, 2, 2.0)
    with pytest.raises(BenchError):
        _ = b - a


def test_a_delta_over_no_reads_reports_zero_rather_than_dividing_by_zero():
    d = DiskDelta("nvme0n1", reads=0, bytes_read=0, ms=0, seconds=1.0)
    assert d.mean_read_bytes == 0.0
    assert d.await_ms == 0.0
    assert d.iops == 0.0


def test_counters_advance_when_the_disk_is_read(datafile):
    dev = device_for(datafile)
    before = disk_counters(dev)
    drop_cache(datafile)
    with open(datafile, "rb") as fh:
        fh.read()
    after = disk_counters(dev)
    assert (after - before).reads >= 0      # monotonic; other load may add


def test_an_unknown_device_is_an_error_not_a_zero():
    with pytest.raises(BenchError):
        disk_counters("nosuchdevice0")


# -- residency ----------------------------------------------------------


def test_reading_a_file_makes_it_resident_and_dropping_it_does_not(datafile):
    warm = warm_cache(datafile)
    assert warm.fraction > 0.9, "a file just read should be in the page cache"
    assert warm.resident_bytes <= datafile.stat().st_size + 4096

    cold = drop_cache(datafile)
    assert cold.fraction < 0.1, "fadvise(DONTNEED) should have evicted it"


def test_residency_can_be_limited_to_a_span(datafile):
    warm_cache(datafile)
    part = residency(datafile, span=MB)
    whole = residency(datafile)
    assert part.total_pages < whole.total_pages
    assert part.resident_pages <= part.total_pages


def test_an_empty_file_has_no_pages(tmp_path):
    p = tmp_path / "empty"
    p.write_bytes(b"")
    r = residency(p)
    assert (r.total_pages, r.resident_pages, r.fraction) == (0, 0, 0.0)


def test_residency_does_not_modify_the_file(datafile):
    before = datafile.read_bytes()
    warm_cache(datafile)
    residency(datafile)
    drop_cache(datafile)
    assert datafile.read_bytes() == before


# -- running commands ---------------------------------------------------


def test_an_unlimited_run_is_a_plain_subprocess():
    e = run_limited([sys.executable, "-c", "print('hi')"])
    assert e.exit_code == 0 and "hi" in e.stdout
    assert e.memory_max_bytes is None and e.peak_memory_bytes is None
    assert not e.timed_out


def test_stderr_is_captured_with_stdout():
    e = run_limited([sys.executable, "-c", "import sys; sys.stderr.write('boom')"])
    assert "boom" in e.stdout


def test_a_run_over_its_budget_is_marked_rather_than_raising():
    e = run_limited([sys.executable, "-c", "import time; time.sleep(30)"], timeout=1.0)
    assert e.timed_out and e.exit_code == -1
    assert e.wall_seconds < 15


def test_a_failing_command_reports_its_code():
    e = run_limited([sys.executable, "-c", "raise SystemExit(3)"])
    assert e.exit_code == 3


def test_asking_for_a_limit_without_the_controller_is_an_error(monkeypatch):
    monkeypatch.setattr("tierinfer.bench.memory_controller_available", lambda: False)
    with pytest.raises(BenchError):
        run_limited([sys.executable, "-c", "pass"], memory_max_bytes=100 * MB)


@pytest.mark.skipif(not memory_controller_available(),
                    reason="memory controller is not delegated to this user")
def test_a_memory_ceiling_binds_and_its_peak_is_reported():
    # Allocate well past the ceiling: the scope must kill it, not the host.
    e = run_limited(
        [sys.executable, "-c", "b = bytearray(400 * 1024 * 1024); print(len(b))"],
        memory_max_bytes=64 * MB, timeout=120)
    assert e.exit_code != 0, "the ceiling did not bind"
    if e.peak_memory_bytes is not None:
        assert e.peak_memory_bytes <= 96 * MB


@pytest.mark.skipif(not memory_controller_available(),
                    reason="memory controller is not delegated to this user")
def test_a_run_inside_its_ceiling_still_succeeds():
    e = run_limited([sys.executable, "-c", "b = bytearray(8 * 1024 * 1024); print('ok')"],
                    memory_max_bytes=256 * MB, timeout=120)
    assert e.exit_code == 0 and "ok" in e.stdout


# -- a whole condition --------------------------------------------------


def test_measure_reports_a_cold_start_and_the_reads_that_followed(datafile, tmp_path):
    m = measure("cold", datafile,
                [sys.executable, "-c", f"open({str(datafile)!r},'rb').read()"],
                cold=True)
    assert m.condition == "cold"
    assert m.execution.exit_code == 0
    assert m.residency_before.fraction < 0.1
    assert m.residency_after.fraction > m.residency_before.fraction
    assert m.disk.seconds > 0

    out = tmp_path / "report.json"
    write_report([m], out)
    import json
    got = json.loads(out.read_text())
    assert got[0]["condition"] == "cold"
    assert "bandwidth_gbps" in got[0]["disk"]
    assert "fraction" in got[0]["residency_after"]


def test_measure_says_so_when_a_cold_start_did_not_take(datafile, monkeypatch):
    monkeypatch.setattr("tierinfer.bench.drop_cache",
                        lambda p: __import__("tierinfer.bench", fromlist=["x"]).Residency(str(p), 100, 100))
    m = measure("cold", datafile, [sys.executable, "-c", "pass"], cold=True)
    assert any("stayed resident" in n for n in m.notes)


def test_a_timed_out_condition_is_labelled_a_lower_bound(datafile):
    m = measure("warm", datafile, [sys.executable, "-c", "import time; time.sleep(30)"],
                timeout=1.0)
    assert any("lower bound" in n for n in m.notes)


@pytest.mark.skipif(not memory_controller_available(),
                    reason="memory controller is not delegated to this user")
def test_the_peak_of_a_real_allocation_is_reported_not_none():
    """The defect this guards: systemd removes the scope before we can ask."""
    e = run_limited(
        [sys.executable, "-c",
         "b = bytearray(300 * 1024 * 1024); b[::4096] = b'x' * (len(b)//4096); print('ok')"],
        memory_max_bytes=1024 * MB, timeout=180)
    assert e.exit_code == 0, e.stdout
    assert e.peak_memory_bytes is not None, "peak was lost with the scope"
    assert e.peak_memory_bytes > 200 * MB, f"peak {e.peak_memory_bytes} is too small to be real"


@pytest.mark.skipif(not memory_controller_available(),
                    reason="memory controller is not delegated to this user")
def test_two_limited_runs_in_the_same_second_do_not_collide():
    """Unit names were pid+second, so two quick runs reused one name."""
    a = run_limited([sys.executable, "-c", "print('a')"], memory_max_bytes=128 * MB, timeout=60)
    b = run_limited([sys.executable, "-c", "print('b')"], memory_max_bytes=128 * MB, timeout=60)
    assert a.exit_code == 0 and b.exit_code == 0, (a.stdout, b.stdout)


def test_a_child_is_given_no_stdin():
    """A runtime that thinks it is interactive waits on a terminal the
    benchmark has not got, and an idle wait that produces a number is worse
    than a failure that does not."""
    e = run_limited([sys.executable, "-c",
                     "import sys; print('closed' if not sys.stdin.read() else 'open')"])
    assert e.exit_code == 0
    assert "closed" in e.stdout


@pytest.mark.skipif(not memory_controller_available(),
                    reason="memory controller is not delegated to this user")
def test_a_run_the_oom_killer_stopped_says_so(datafile):
    """A row of dashes reads as 'produced no number'; this produced none for
    a reason, and the reason is the result."""
    m = measure("cold+tiny", datafile,
                [sys.executable, "-c", "b = bytearray(2 * 1024 * 1024 * 1024); print(len(b))"],
                memory_max_bytes=32 * MB, timeout=180)
    assert m.execution.exit_code != 0
    assert any("OOM killer" in n for n in m.notes), m.notes


def test_a_nonzero_exit_is_flagged_rather_than_passed_over(datafile):
    m = measure("warm", datafile, [sys.executable, "-c", "raise SystemExit(4)"])
    assert any("exited 4" in n for n in m.notes), m.notes


def test_dontneed_cannot_evict_pages_another_process_has_mapped(tmp_path):
    """The mechanism that decides TierInfer's architecture.

    posix_fadvise(DONTNEED) drops clean page-cache pages — but not ones a live
    process holds mapped, because the mapping keeps a reference. So residency
    of a model file cannot be managed from beside the runtime that mmapped it,
    in either direction: WILLNEED has nowhere to read into under a full
    cgroup, and DONTNEED cannot release what the mapping is holding.

    Measured end to end this cost a 16-token run 4 seconds and changed
    residency by nothing. Here it is in isolation, in under a minute.
    """
    p = tmp_path / "blob.bin"
    p.write_bytes(os.urandom(64 * MB))
    assert drop_cache(p).fraction < 0.1

    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import mmap, os, sys, time\n"
         f"fd = os.open({str(p)!r}, os.O_RDONLY)\n"
         "mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)\n"
         "n = 0\n"
         "for off in range(0, mm.size(), mmap.PAGESIZE): n += mm[off]\n"
         "print('touched', flush=True)\n"
         "time.sleep(60)\n"],
        stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "touched"
        assert residency(p).fraction > 0.9, "the holder did not fault the file in"
        # The advice is issued and returns success; it simply does nothing.
        assert drop_cache(p).fraction > 0.9, \
            "DONTNEED evicted mapped pages — this platform behaves differently " \
            "and the architecture conclusion drawn from it needs revisiting"
    finally:
        holder.terminate()
        holder.wait(timeout=10)

    # With the mapping gone, the same call works.
    assert drop_cache(p).fraction < 0.1
