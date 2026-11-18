// Gate G-6 (lane O, O2): the int-only rectifier DP that O2 deletes,
// src/utils/flip_zero.cpp at the base, against the templated copy the
// library uses, src/bc/flip_zero_impl.h<int> at the step, on every binary
// sequence of length 1..14, minRun0 and minRun1 in 1..3, both flip modes and
// stride 1 and 2. Equal means the same minimum flip count and, when it is
// possible, the same fixed sequence. The new copy with dir = -1 on the
// reversed sequence must give the same minimum count.
//
// g6_flip_zero.py extracts the two files next to this one (old_flip_zero.cpp,
// flip_zero_impl.h) with a configure.h that defines DISPATCH_MACRO as the
// library does. Compiled as C++ the cases run on the host; compiled by nvcc
// (-x cu) both copies are __host__ __device__ and every case runs in a CUDA
// kernel, one thread per case, so the CUDA leg compares device results only.

// the headers the two copies include, outside the namespaces below
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "configure.h"

namespace oldimpl {
#include "old_flip_zero.cpp"  // namespace snap { int-only copy }; #undefs its macros
}
namespace newimpl {
#include "flip_zero_impl.h"  // namespace snap { template copy }; #undefs its macros
}

constexpr int kNmax = 14, kSmax = 2, kRun = 4;  // MAXRUN = 4 in both copies
constexpr int kDp = (kNmax + 1) * kSmax * kRun * 2;

struct Case {
  int n, r0, r1, both, stride, bits;
};

// flags: 1 = differs (count or sequence), 2 = impossible, 4 = reversed
// count differs (stride 1 only)
DISPATCH_MACRO int run_case(Case c) {
  int in[kNmax * kSmax], a[kNmax * kSmax], b[kNmax * kSmax];
  int dp[kDp], fl[kDp], fb[kDp], uf[kDp];
  for (int i = 0; i < c.n * c.stride; ++i) in[i] = a[i] = b[i] = 7;
  for (int i = 0; i < c.n; ++i) in[i * c.stride] = (c.bits >> i) & 1;
  int flags = 0;

  int ma = oldimpl::snap::compute_min_flips(in, c.n, c.r0, c.r1, c.both,
                                            c.stride, dp, fl, fb, uf);
  if (ma >= 0)
    oldimpl::snap::reconstruct_solution(a, in, c.n, c.r0, c.r1, c.both,
                                        c.stride, fl, fb, uf);

  for (int i = 0; i < c.n * c.stride; ++i) b[i] = in[i];
  int mb = newimpl::snap::compute_min_flips<int>(in, c.n, c.r0, c.r1, c.both,
                                                 1, c.stride, dp, fl, fb, uf);
  if (mb >= 0)
    newimpl::snap::reconstruct_solution<int>(b, c.n, c.r0, c.r1, c.both, 1,
                                             c.stride, fl, fb, uf);

  if (ma != mb) flags |= 1;
  if (ma >= 0 && ma == mb)
    for (int i = 0; i < c.n; ++i)
      if (a[i * c.stride] != b[i * c.stride]) flags |= 1;
  if (ma < 0) flags |= 2;

  if (c.stride == 1) {
    int rin[kNmax];
    for (int i = 0; i < c.n; ++i) rin[i] = in[c.n - 1 - i];
    int mr = newimpl::snap::compute_min_flips<int>(rin, c.n, c.r0, c.r1, c.both,
                                                   -1, 1, dp, fl, fb, uf);
    if (mr != ma) flags |= 4;
  }
  return flags;
}

#ifdef __CUDACC__
__global__ void kernel(Case const* cases, int* flags, long ncase) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i < ncase) flags[i] = run_case(cases[i]);
}
#define CK(x)                                                         \
  do {                                                                \
    cudaError_t e = (x);                                              \
    if (e != cudaSuccess) {                                           \
      std::fprintf(stderr, "CUDA: %s (%s)\n", cudaGetErrorString(e), #x); \
      std::exit(3);                                                   \
    }                                                                 \
  } while (0)
#endif

int main() {
  std::vector<Case> cases;
  for (int stride = 1; stride <= kSmax; ++stride)
    for (int n = 1; n <= kNmax; ++n)
      for (int r0 = 1; r0 <= 3; ++r0)
        for (int r1 = 1; r1 <= 3; ++r1)
          for (int both = 0; both <= 1; ++both)
            for (int bits = 0; bits < (1 << n); ++bits)
              cases.push_back({n, r0, r1, both, stride, bits});
  long ncase = static_cast<long>(cases.size());
  std::vector<int> flags(ncase);

#ifdef __CUDACC__
  Case* dcases;
  int* dflags;
  CK(cudaMalloc(&dcases, ncase * sizeof(Case)));
  CK(cudaMalloc(&dflags, ncase * sizeof(int)));
  CK(cudaMemcpy(dcases, cases.data(), ncase * sizeof(Case),
                cudaMemcpyHostToDevice));
  int threads = 128;
  kernel<<<(ncase + threads - 1) / threads, threads>>>(dcases, dflags, ncase);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(flags.data(), dflags, ncase * sizeof(int),
                cudaMemcpyDeviceToHost));
  CK(cudaFree(dcases));
  CK(cudaFree(dflags));
  char const* where = "cuda";
#else
  for (long i = 0; i < ncase; ++i) flags[i] = run_case(cases[i]);
  char const* where = "cpu";
#endif

  long mism = 0, impossible = 0, rev = 0, rev_mism = 0, shown = 0;
  for (long i = 0; i < ncase; ++i) {
    if (flags[i] & 1) {
      ++mism;
      if (shown++ < 5) {
        Case const& c = cases[i];
        std::printf("MISMATCH n=%d bits=%d r0=%d r1=%d both=%d stride=%d\n",
                    c.n, c.bits, c.r0, c.r1, c.both, c.stride);
      }
    }
    if (flags[i] & 2) ++impossible;
    if (cases[i].stride == 1) ++rev;
    if (flags[i] & 4) ++rev_mism;
  }
  std::printf("device: %s\n", where);
  std::printf("dir=+1: %ld cases (%ld impossible), %ld mismatches (count and "
              "sequence)\n",
              ncase, impossible, mism);
  std::printf("dir=-1 on reversed input: %ld cases, %ld min-flip mismatches\n",
              rev, rev_mism);
  std::printf("G-6 %s\n", mism == 0 && rev_mism == 0 ? "PASS" : "FAIL");
  return mism == 0 && rev_mism == 0 ? 0 : 1;
}
