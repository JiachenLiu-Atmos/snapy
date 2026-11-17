# Straka mixed-precision diagnosis against Kratos

The controlled installed-binary GPU comparison gives 63.4482 s FP64 versus
70.2006 s FP32 (three sequential runs per precision on the same RTX 5090;
10.6423% greater FP32 latency). All six runs passed. These are pre-optimization
binaries. Repository copies of the raw timing data and hashes have been removed.
The recent face-state/scratch changes have CPU validation but no optimized GPU
measurement. This analysis does not conflate those versions.

## Reference method

Lile Wang, *The Kratos Framework for Heterogeneous Astrophysical Simulations:
Fundamental Infrastructures and Hydrodynamics*, ApJS 277, 63 (2025),
[DOI](https://doi.org/10.3847/1538-4365/adbdb6),
[author manuscript, §§3.2–3.3](https://arxiv.org/html/2501.02317v2#S3.SS2).
The publisher PDF did not load through the browser; its indexed text and the
March 2025 author manuscript were consulted.

Kratos keeps conserved quantities and retained integration stages in FP64,
uses FP32 reconstruction/Riemann work, and organizes flux computation around
shared-memory reuse. It avoids subtracting close wave and flow speeds by
forming their acoustic differences directly. Its small-amplitude tests also
show limitations of reduced-precision reconstruction. Its PLM/Heun benchmarks
are not the same algorithms or workload as this WENO5/RK3 Straka experiment.

## Confirmed source/binary issue: FP32 tensors do not imply FP32 arithmetic

In the pre-fix `src/riemann/hllc_impl.h:41`, `auto rhoa = .5 * (...)` deduces double even
for T=float; the same occurs for `ca`, `pmid`, shock factors and downstream wave
speed algebra. Other unsuffixed floating constants and the double sqrt overload
continue that promotion. This is separate from intentional FP64 background
pressure cancellation in the hydro wrapper.

Disassembled the installed SM120 native library without compiling:

```sh
/usr/local/cuda-13.1/bin/cuobjdump --dump-sass --gpu-architecture sm_120 \
  /home/chengcli/pyenv/lib64/python3.11/site-packages/snapy/lib/libsnap_cuda_release.so \
  > /tmp/snapy-m6-hsaZRT/results/snapy-sm120.sass
```

The N=8 float HLLC symbol matched the profiler's float specialization. Static
instruction occurrences include 101 DFMA, 50 DMUL, 15 DADD and 59 F32→F64
conversions. These are static code counts, not executed-instruction counts or a
cycle attribution. The repository copy of the disassembly summary has been removed.
HLLC float GPU duration is 0.2125 ms versus double 0.2102 ms over the five-step
profile: this kernel currently supplies no measured speed advantage.

## Dominant small-case evidence: fragmented work and conversions

Kernel durations below sum only Chrome-trace events with category `kernel`;
CPU operator self-device times are excluded to avoid counting the same GPU
work twice. These are profile samples, not end-to-end benchmark times.

| Five-step quantity | FP64 | FP32 |
| --- | ---: | ---: |
| GPU kernel launches | 8665 | 9765 |
| Summed GPU kernel duration (ms) | 13.6384 | 14.9680 |
| HLLC GPU duration (ms) | 0.2102 | 0.2125 |
| Main WENO stencil-kernel duration (ms) | 1.4739 | 0.7345 |
| CPU self-time in cudaLaunchKernel (ms) | 18.5516 | 23.5314 |
| aten::_to_copy calls | 105 | 805 |

HLLC is approximately 1.4% of FP32 summed GPU kernel duration. Even eliminating
it would not fix the overall slowdown on this grid. The WENO kernel already
benefits from FP32, but surrounding work loses more than it saves. Each step
launches about 1733 versus 1953 kernels for only 16,384 active cells. Average
kernel durations are roughly 1.5–1.6 microseconds. This strongly supports a
launch/dispatch and conversion-overhead diagnosis; it does not constitute a
roofline analysis or prove a particular memory-bandwidth bottleneck.

Snapy already uses custom CUDA kernels and shared-memory stencil code (see
`src/recon/interp_impl.cuh`), so replacing all PyTorch operations or claiming
that no shared memory is used would be incorrect. The missing opportunity is
reducing intermediate tensors and launches *between* these kernels. Existing
profiles retain substantial FP64 elementwise arithmetic as well as casts.
There is no moist chemistry step in this dry Straka experiment.

## Precision policy and numerical stability

Snapy's FP32 mode stores evolving anomalies and RK state in FP32, with a static
FP64 reference. Thus its accumulated update rounding is FP32 despite retaining
the background. Changing only reconstruction/forcing scratch cannot change this
conservation contract. This distinction concerns long-time numerical accuracy;
it does not by itself explain the measured runtime penalty.

The pre-fix HLLC calculates `al = uL - cL*qL`, then `vxl = uL - al` for the
contact-speed formula. Once accidental double promotion is removed, use the
algebraically equivalent `vxl = cL*qL` and `vxr = -cR*qR` there. The later flux
formula uses *clipped* wave speeds `bm/bp`; those differences need branch-aware
handling, not an unconditional replacement. Preserve shifted-pressure formulas
and test boosted flows, shocks, rarefactions and near-equal states. Straka's
low-Mach pressure cancellation remains a separate FP64-sensitive operation.

## Revised priorities

1. Audit scalar types inside FP32 kernels, starting with HLLC. Use typed
   constants/intermediates, verify generated instructions, and combine the
   change with cancellation-safe algebra. Compare against FP64 over a broad
   state range; require flux/positivity/conservation and Straka regressions.
   Do not merely replace every double literal indiscriminately.
2. Retain the face-row/scratch improvements, then fuse the pressure/background
   transformations, dry face EOS algebra and HLLC preparation where practical.
   Reuse face data inside kernels; avoid writing whole intermediate fields to
   global memory just to cast/read them again. Profile each change separately.
3. Add an experimental FP64 evolving-anomaly/RK accumulation plus FP32 compute
   policy, separate from the existing FP32-storage mode. Accumulate shared-face
   flux divergence and sources in FP64. This must preserve hydrostatic reference
   cancellation, scalar/species conventions and restart dtype contracts. It is
   an A/B accuracy/performance experiment, not an assumed speedup.
4. Repeat same-GPU timing on the optimized binary, then sweep grid size with
   matched physics/time integration to distinguish small-grid launch overhead
   from arithmetic/memory limits. Measure end-to-end and kernel-only intervals.

No additional native changes or GPU recompilation were made for this diagnosis.
No kintera source/API change is presently indicated. A new precision policy would
need explicit compatibility testing at existing kintera and restart boundaries.

The subsequent HLLC fix removes these implicit promotions in the shared CPU/CUDA
kernel and adds safe float acoustic differences; see the [speed plan](fp32-straka-speed-plan.md).
The installed GPU binary and all GPU measurements above remain pre-fix evidence.
