# Patches to other runtimes

TierInfer changes nothing in llama.cpp (the shim is preloaded). FreeToken
needs a small patch to hand its CPU-executor layers' banks to TierInfer;
the upstream (FlashML-org/FreeToken) is not ours to push to, so the patch
is kept here and applied to a checkout:

    cd ~/freetoken-qwen38 && git checkout -b tierinfer-tier && git am <TierInfer>/patches/freetoken-tierinfer-tier.patch

`freetoken-tierinfer-tier.patch` — `HostResidency.TIERED`,
`HostBank(backing="tierinfer")`, FTW load without the fill for tiered
layers, engine labels from `TIERINFER_SOCK` + `--moe-cpu-layers`, and the
CPU executor's per-layer routing log reported as `ROUTED`. See
`docs/FREETOKEN.md`. Made against FreeToken `9535656`.
