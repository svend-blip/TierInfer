# TierInfer — slutrapport, 2026-09-18

Alt herunder er målt på denne maskine (RTX 5090 32 GB, 187 GB RAM, md0
RAID0 over to USB-4M2-SSD'er på `/data/ai-data`), kold page-cache før hver
kørsel, greedy tokens sammenlignet på tværs af arme. Kilder: CP-1…CP-16 i
`docs/CHECKPOINTS.md`, `benchmarks/480b/VALIDATION-480B.md`, `benchmarks/*-out/`.

## Status

120 acceptance-rækker: **100 VERIFIED, 19 ACCEPTED, 1 BLOCKED** (TI-LLAMA-008:
VRAM-tier under llama.cpp kræver en llama.cpp-patch; llama.cpp's egen
`-ncmoe` ejer kortet). Alle 15 oprindelige mål og addendummets 20 kriterier er
lukket med evidens; negative resultater er registreret som målt.

## 1. llama.cpp + TierInfer, Qwen3-Coder-480B-A35B Q4_K_M (270 GiB, 6 shards)

`-ngl 99 -ncmoe 60 -fa on`, 111-token prompt, 32 tokens, tre kolde runs pr. arm.

| arm | load | prompt t/s | **gen t/s** | inferens-I/O | reads | gns. read |
|---|--:|--:|--:|--:|--:|--:|
| native | 192 s | 0,767 | **0,467** (0,437–0,481) | 246 GiB | 2,26 M | 107–116 KB |
| loader, 150 GB tier | 37 s | 0,750 | **0,647** (0,622–0,668) | 158 GiB | 427 k | 387 KB |

Generering **+39 %** (medianer; dårligste loader-run 29 % over bedste native),
36 % færre bytes, 5,3× færre requests, tokens identiske i alle seks runs.
Pr. genereret token: 93,6 % hit i tier'en, 832 faults, 454 MB kopieret,
16 evictions. Prompt-batchen fylder tier'en: 5 100 misses, 150 GB på 148 s
(1,0 GB/s = enhedens loft for ekspert-store læsninger).

Native `-ngl 0` (CP-4): 0,241 t/s kold median, 1,82 GB / 74 k reads à 24 KB
pr. token, CPU-bundet page-faulting med enheden halvt ledig.

## 2. Replay (ægte routing, ægte filer, ingen compute), 480B

| RAM-tier | hit | bytes/token | reads/token |
|--:|--:|--:|--:|
| 100 GiB | 86,8 % | 1,30 GB | 3 771 |
| 150 GiB | 91,7 % | 0,63 GB | 1 815 (à 361 KB) |

Mod native's 1,82 GB / 74 k: 36–65 % færre bytes, 20–40× færre operationer,
med mindre RAM end native's page-cache. VRAM-tier: 22,1 % hit fra 792 slots.
Prefetch: ±0. Fejlinjektioner: 17 856 leveringer, 0 mismatches.

## 3. FreeToken + TierInfer, Qwen3.8 Flash-Next NVFP4 (121 GB FTW), 12 af 48 MoE-lag på CPU-executor

95 prompt + 63 tokens; 17,1 GB banker i de 12 lag.

| arm | load | wall | FreeToken decode | inferens-I/O |
|---|--:|--:|--:|--:|
| native (alt resident) | 55 s | 4,3 s | 34,8–35,0 t/s | 0,01 GiB |
| tiered, 16 GB budget | 44 s | 25 s | **35,4 / 34,8 t/s** | 15,5 GiB |
| tiered, 8 GB budget | 44 s | 38–40 s | 5,4–6,3 t/s | 18,7 GiB / 83 k reads à 235 KB |
| tiered, 8 GB + prefetch dybde 8 | 44 s | 38 s | 6,2–6,3 t/s | 18,6 GiB |

Holder budgettet lagene, er decode = native (0 misses, 27 ms/skridt); prisen
er prefill, der trækker 17,1 GB gennem tier'en (~23 s). Ved halvt budget:
85,8 % hit pr. skridt, 42 MB og 212–246 ms pr. skridt. Output identisk i 11
runs. Prefetch: 48 gæt på 62 skridt, 14 nyttige, ingen effekt.

**Servertab under kørsel:** `tierinfer serve` dræbt 12 s inde i en completion;
klienten serverede selv resten fra checkpointet; 63 tokens identiske med
native; 42 s mod 25 s uforstyrret.

## 4. Predictor, prefetch, align (GLM-4.5-Air IQ4_XS, `-ngl 0`, 27 GB tier, to runs pr. arm)

| arm | gen t/s | md0 reads | gns. | prefetch udstedt/nyttig/sen/spildt |
|---|--:|--:|--:|---|
| eksakt, dybde 0 | 0,340 / 0,382 | 191 k | 375 KB | — |
| align 512 KiB, dybde 0 | 0,390 / 0,384 | 172 k | 439 KB | — |
| dybde 8, adaptiv blend | 0,340 / 0,355 | 197 k | 374 KB | 411 / 304 / 140 / 98 |
| dybde 8, prerouter online | 0,193 / 0,194 | 202 k | 374 KB | 2 110 / 818 / 148 / 726 |

Prerouter offline (480B-traces): recall@16 online 77,4 % prose / 66,3 % kode
mod blendens 67,6 / 57,3 (+10 pt i alle retninger, også på tværs af
promptklasser). Live halverer den genereringen: flere gæt, mere I/O på en
mættet enhed. Align: 10 % færre requests, 15 % flere bytes, t/s inden for
støjen. Dybde 8: ingen gevinst. Defaults: demand-only, eksakte reads.

## 5. GLM første live A/B (CP-11) og FlowRunner

GLM loader vs native under 32 GB cgroup: 0,324 vs 0,310 t/s, identiske
tokens (loader-runs 2–3 stod ved siden af pga. stale socket, rettet).
FlowRunner kørte begge runtimes gennem motorgrænsen: GLM (startup 21 s,
0,389 t/s, telemetri læst tilbage) og FreeToken (startup 44,5 s, 31 tokens
på 24,6 s).

## 6. Defekter fundet af tallene (alle rettet)

Eviction i unmappet hukommelse (llama.cpp's suffix-fragment → `free():
invalid pointer`); ROUTED-burst scoret mod tømt fault-sæt; FreeToken's
routing-log som pinned→pinned `copy_` (ikke i CUDA-grafen); stream ikke
drænet før log-læsning; sovende tråde efter servertab (UFFDIO_WAKE);
FlowRunner så ikke zombie-processer (Signal(0)); runtime uden eget
bin-dir/CUDA på PATH; stale socket-fil lod shim'en stå ved siden af; 36 k
"repairs" pr. 30 tokens før pagemap-tjekket; prio-stale prefetch-jobs.
