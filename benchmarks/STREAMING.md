# Streaming — issuing expert reads instead of waiting for them

Measured 2026-09-17, same host and model as `BASELINE.md`, with the device
otherwise idle.

```
python benchmarks/streaming.py MODEL.gguf --layers 8 --experts 16 \
    --workers 2 --workers 4 --workers 8 --workers 16 --workers 32
```

128 experts (8 layers × 16), 1,210 MB in 384 byte ranges, 9.97 MB per expert.
Each method is measured with those ranges evicted from the page cache first,
so none of them is answered out of RAM.

| method | seconds | MB/s | vs demand paging |
|--------|--------:|-----:|-----------------:|
| mmap-fault |  1.38 |  879 | 1.00 |
| pread-sync |  1.99 |  608 | 0.69 |
| stream-2   |  0.92 | 1,313 | 1.49 |
| stream-4   |  0.54 | 2,224 | 2.53 |
| **stream-8** | **0.40** | **3,047** | **3.47** |
| stream-16  |  0.41 | 2,978 | 3.39 |
| stream-32  |  0.50 | 2,400 | 2.73 |

All seven returned byte-identical data (`blake2b` digest
`9934a447f996016467f603615af947e7`). A faster reader that returns different
bytes is not a faster reader, so the benchmark fails loudly rather than
printing a ranking if the digests disagree.

## What this says

**Concurrency is the whole difference, not the syscall.** A single-threaded
`pread` is *slower* than letting the kernel fault the pages in — 608 against
879 MB/s. Replacing demand paging with explicit reads buys nothing on its
own; mmap's readahead is good at what it does. What buys something is having
eight of them outstanding at once, which mmap faulting cannot do from one
thread because each fault blocks it.

**Eight workers is this device's shape.** 3,047 MB/s at eight, essentially
flat at sixteen, and worse at thirty-two. Past the point where the queue is
deep enough to keep the device busy, more threads only add contention. The
number is a property of the NVMe and the host, not of the code, so it is
measured rather than chosen.

**It beats what the runtime achieves for itself.** The cold baseline measured
llama.cpp loading the model at 1.54 GB/s. Explicit streaming at eight workers
reaches 3.0 GB/s on the same device — about twice as fast at getting the same
kind of bytes off the same disk.

## What it does not say

**These are cold reads of a warm-ish file layout.** Each expert is one
contiguous slab per projection, and the eight layers are far apart in the
file, so this is scattered at the layer scale and sequential within an
expert. A layout that interleaved experts would read differently.

**No compute overlaps it.** This measures the read path alone. Whether the
lead time is enough to hide a stall depends on how long a layer takes to
compute, which is goal 7's question and needs goal 6 to answer.

**Python's threads are enough here only because `pread` releases the GIL.**
That holds for the read itself; it would not hold if the bytes needed
processing on the way in.
