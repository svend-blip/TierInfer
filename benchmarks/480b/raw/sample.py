#!/usr/bin/env python3
"""Sample /proc/diskstats, /proc/meminfo, /proc/stat and nvidia-smi every N s to CSV.
usage: sample.py OUT.csv INTERVAL dev1 dev2 ...   (stop with SIGTERM)"""
import sys, time, signal, subprocess
out, interval, devs = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
stop = False
signal.signal(signal.SIGTERM, lambda *a: globals().__setitem__('stop', True))
def diskstats():
    d = {}
    for line in open('/proc/diskstats'):
        f = line.split()
        if len(f) >= 14 and f[2] in devs:
            # reads, read_merges, sectors, ms, in_flight, io_ticks
            d[f[2]] = (int(f[3]), int(f[4]), int(f[5]), int(f[6]), int(f[11]), int(f[12]))
    return d
def meminfo():
    m = {}
    for line in open('/proc/meminfo'):
        k, v = line.split(':'); m[k] = int(v.split()[0]) * 1024
    return m
def cpu():
    f = open('/proc/stat').readline().split()
    v = list(map(int, f[1:8])); return sum(v), v[3] + v[4]  # total, idle+iowait
def vram():
    try:
        return int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=3).stdout.split()[0]) * 1024 * 1024
    except Exception:
        return -1
with open(out, 'w', buffering=1) as fh:
    cols = ['t', 'mem_available', 'cached', 'vram_used', 'cpu_busy_frac']
    for d in devs:
        cols += [f'{d}_rps', f'{d}_merges_ps', f'{d}_MBps', f'{d}_avg_kb', f'{d}_await_ms', f'{d}_util', f'{d}_inflight']
    fh.write(','.join(cols) + '\n')
    p_d, p_c, p_t = diskstats(), cpu(), time.time()
    while not stop:
        time.sleep(interval)
        d, c, t = diskstats(), cpu(), time.time()
        dt = t - p_t; m = meminfo()
        ct, ci = c[0] - p_c[0], c[1] - p_c[1]
        row = [f'{t:.1f}', str(m['MemAvailable']), str(m['Cached']), str(vram()), f'{(ct - ci) / ct if ct else 0:.3f}']
        for dev in devs:
            a, b = p_d[dev], d[dev]
            r, mg, sec, ms, infl, tick = (b[i] - a[i] for i in range(6))
            infl = b[4]
            row += [f'{r/dt:.0f}', f'{mg/dt:.0f}', f'{sec*512/dt/1e6:.1f}', f'{sec*512/r/1024:.1f}' if r else '0',
                    f'{ms/r:.2f}' if r else '0', f'{tick/dt/1000:.2f}', str(infl)]
        fh.write(','.join(row) + '\n')
        p_d, p_c, p_t = d, c, t
