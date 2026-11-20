// C/C++
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// torch
#include <torch/torch.h>

#include "cuda_test_gate.hpp"

// snap
#include <snap/bc/internal_boundary.hpp>

// int-only DP of the rectifier; the library instantiates the same template
// (bc/bc_dispatch.cpp) on the solid mask's integral dtype
#include <snap/bc/flip_zero_impl.h>

using namespace snap;

// Ported from the int-only copy src/utils/flip_zero.{cpp,h} (deleted in O2)
// to bc/flip_zero_impl.h<int> (gate G-6). Each row: input, minRun0,
// minRun1, allowBothFlips, minimum flips (-1: impossible) and the fixed
// sequence, as the deleted copy produced them at 730b3500. Exhaustive
// equality of the two implementations (n <= 14, every run length and flip
// mode, stride 1 and 2) was checked before the deletion.
int test1() {
  struct Case {
    char const *in;
    int min_run0, min_run1, allow_both;
    int flips;
    char const *out;
  } const cases[] = {
      {"0100110", 3, 2, 0, 4, "1111111"},
      {"0100110", 3, 2, 1, 2, "0000111"},
      {"1011101", 2, 2, 0, 2, "1111111"},
      {"000100010000", 3, 2, 0, 3, "000111110000"},
      {"110010011", 2, 3, 1, 3, "111000111"},
      {"0101010101", 2, 2, 0, 5, "1111111111"},
      {"0101010101", 3, 3, 1, 4, "0001110000"},
      {"1", 1, 2, 0, -1, ""},
      {"10", 2, 2, 0, 1, "11"},
      {"0000", 3, 2, 0, 0, "0000"},
      {"1101100111010", 3, 2, 0, 5, "1111111111111"},
      {"0010111010011", 2, 3, 1, 3, "0000111000111"},
  };

  constexpr int MAXRUN = InternalBoundaryOptionsImpl::MAXRUN;
  int nfail = 0;
  for (auto const &c : cases) {
    int n = strlen(c.in);
    std::vector<int> seq(n), dp((n + 1) * MAXRUN * 2),
        fromLen((n + 1) * MAXRUN * 2), fromBit((n + 1) * MAXRUN * 2),
        usedFlip((n + 1) * MAXRUN * 2);
    for (int i = 0; i < n; i++) seq[i] = c.in[i] - '0';

    int dir = 1, stride = 1;
    int flips = compute_min_flips<int>(
        seq.data(), n, c.min_run0, c.min_run1, c.allow_both, dir, stride,
        dp.data(), fromLen.data(), fromBit.data(), usedFlip.data());

    std::string out;
    if (flips >= 0) {
      // the bc copy writes the fixed sequence in place
      reconstruct_solution<int>(seq.data(), n, c.min_run0, c.min_run1,
                                c.allow_both, dir, stride, fromLen.data(),
                                fromBit.data(), usedFlip.data());
      for (int i = 0; i < n; i++) out += static_cast<char>('0' + seq[i]);
    }

    if (flips != c.flips || out != c.out) {
      fprintf(stderr,
              "rectify %s (%d, %d, %d): flips %d, expected %d; %s, "
              "expected %s\n",
              c.in, c.min_run0, c.min_run1, c.allow_both, flips, c.flips,
              out.c_str(), c.out);
      ++nfail;
    }
  }
  return nfail;
}

int test2() {
  int flips = 0;
  auto solid = torch::zeros({8, 8, 8}, torch::kInt32);
  if (snapy_cuda_test_enabled()) solid = solid.to(torch::kCUDA);

  solid[4][4][4] = 1;
  solid[2][5][3] = 1;

  // std::cout << "solid = " << solid << std::endl;

  auto op = InternalBoundaryOptionsImpl::create();
  InternalBoundary pib(op);

  auto out = pib->rectify_solid(solid, flips);
  std::cout << "out = " << out << std::endl;
  std::cout << "flips = " << flips << std::endl;

  // every ghost slab is solid, the outer ones included (nghost = 1 without a
  // MeshBlock; only 0 -> 1 flips are allowed, so the DP cannot undo it)
  int nghost = 1;
  for (int dim = 0; dim < 3; ++dim) {
    auto lo = out.narrow(dim, 0, nghost);
    auto hi = out.narrow(dim, out.size(dim) - nghost, nghost);
    if (!lo.eq(1).all().item<bool>() || !hi.eq(1).all().item<bool>()) {
      std::cerr << "ghost slab along dim " << dim << " is not solid: inner "
                << lo.eq(1).all().item<bool>() << " outer "
                << hi.eq(1).all().item<bool>() << std::endl;
      return 1;
    }
  }
  return 0;
}

int main(int argc, char *argv[]) {
  int nfail = test1();
  return test2() + nfail;
}
