// A Mesh's block worker pool gives every block a CUDA stream of its own.
//
// Its blocks launch, capture and replay on their worker streams. A Cartesian
// slab Mesh of 33 and of 40 blocks in one process: the pool has one stream per
// block, no two blocks share one, and none is a stream of torch's shared pool
// (32 per priority per device, handed out in turn), from which a 33rd block
// would get a stream another block already launches on.
#include <gtest/gtest.h>
#include <snap/snap.h>
#include <torch/torch.h>
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <snap/mesh/mesh.hpp>
#include <string>
#include <vector>

#include "cuda_test_gate.hpp"

#ifdef USE_CUDA
#include <c10/cuda/CUDAStream.h>
#endif

using namespace snap;

namespace {

Mesh make_mesh(int nblocks) {
  auto tmpl = (std::filesystem::temp_directory_path() /
               "test-mesh-worker-streams-XXXXXX")
                  .string();
  std::vector<char> path(tmpl.begin(), tmpl.end());
  path.push_back('\0');
  char* f = path.data();
  close(mkstemp(f));
  std::ofstream(f)
      << "geometry: {type: cartesian, bounds: {x1min: 0., x1max: 1., x2min: "
         "0., x2max: 1., x3min: 0., x3max: 1.}, cells: {nx1: 4, nx2: "
      << 4 * nblocks << ", nx3: 1, nghost: 2}}\n"
      << "distribute: {layout: slab, nb2: " << nblocks
      << ", nb3: 1, verbose: false}\n"
      << "dynamics: {equation-of-state: {type: ideal-gas}}\n"
      << "boundary-condition: {external: {x1-inner: reflecting, x1-outer: "
         "reflecting, x2-inner: periodic, x2-outer: periodic, x3-inner: "
         "periodic, x3-outer: periodic}}\n"
      << "integration: {type: rk3, cfl: 0.4, implicit-scheme: 0}\n";
  auto block_opts = MeshBlockOptionsImpl::from_yaml(f);
  std::remove(f);
  block_opts->layout()->device("cuda");
  block_opts->layout()->device_id(0);
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(nblocks);
  return Mesh(mesh_opts);
}

}  // namespace

TEST(MeshWorkerStreams, no_two_blocks_share_a_stream) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";
#ifdef USE_CUDA
  // every stream of the shared pool: it hands out its 32 per priority in turn
  std::set<void*> pooled;
  for (int i = 0; i < 64; ++i)
    pooled.insert(c10::cuda::getStreamFromPool(i % 2 == 1, 0).stream());
  for (int nblocks : {33, 40}) {
    auto mesh = make_mesh(nblocks);
    auto streams = mesh->worker_streams();
    ASSERT_EQ(streams.size(), static_cast<size_t>(nblocks));
    std::set<void*> distinct(streams.begin(), streams.end());
    EXPECT_EQ(distinct.size(), streams.size())
        << nblocks << " blocks share " << streams.size() - distinct.size()
        << " worker streams";
    int from_pool = 0;
    for (auto* s : streams) from_pool += pooled.count(s);
    EXPECT_EQ(from_pool, 0) << nblocks << " blocks: " << from_pool
                            << " worker streams come from the shared pool";
  }
#endif
}
