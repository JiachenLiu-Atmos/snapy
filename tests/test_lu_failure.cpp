#include <gtest/gtest.h>
#include <snap/implicit/forward_sweep_impl.h>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <fstream>
#include <limits>
#include <snap/mesh/mesh.hpp>
#include <snap/mesh/meshblock.hpp>

#include "cuda_test_gate.hpp"

namespace {

template <typename T, int N>
void factor_cases() {
  using Mat = Eigen::Matrix<T, N, N, Eigen::RowMajor>;
  int ix[N];
  Mat a = Mat::Identity();
  EXPECT_EQ(ludcmp(a, ix), 1);
  a.setIdentity();
  a.row(0).swap(a.row(1));
  EXPECT_EQ(ludcmp(a, ix), -1);
  for (T scale : {T(1.e-10), T(1), T(1.e10)}) {
    a.setIdentity();
    a *= scale;
    EXPECT_NE(ludcmp(a, ix), 0);
    a.setIdentity();
    a.row(1) = a.row(0);
    a(1, 1) = T(4) * std::numeric_limits<T>::epsilon();
    a *= scale;
    EXPECT_EQ(ludcmp(a, ix), 0) << "near-singular row scale " << scale;
  }
  a.setIdentity();
  a.row(0).setZero();
  EXPECT_EQ(ludcmp(a, ix), 0);
  a.setIdentity();
  a.row(N - 1) = a.row(0);
  EXPECT_EQ(ludcmp(a, ix), 0) << "final zero pivot";
  for (T bad : {std::numeric_limits<T>::quiet_NaN(),
                std::numeric_limits<T>::infinity()}) {
    a.setIdentity();
    a(0, N - 1) = bad;
    EXPECT_EQ(ludcmp(a, ix), 0);
  }
  // Finite inputs whose elimination overflows must also be refused.
  a.setIdentity();
  T large = std::numeric_limits<T>::max();
  a(0, 0) = large;
  a(0, 1) = large;
  a(1, 0) = -large;
  a(1, 1) = large;
  EXPECT_EQ(ludcmp(a, ix), 0);
}

template <typename T, int N>
void sweep_cases() {
  using Mat = Eigen::Matrix<T, N, N>;
  using Vec = Eigen::Matrix<T, N, 1>;
  for (int badcell = 0; badcell < 2; ++badcell) {
    for (int kind = 0; kind < 5; ++kind) {
      Mat a[2], b[2], c[2];
      Vec delta[2];
      T du[10];
      for (auto &x : du) x = 1;
      for (int i = 0; i < 2; ++i) {
        a[i].setIdentity();
        b[i].setZero();
        c[i].setZero();
        delta[i].setZero();
      }
      if (kind == 1 || kind == 2) {
        a[badcell].row(1) = a[badcell].row(0);
        if (kind == 2)
          a[badcell](1, 1) = T(4) * std::numeric_limits<T>::epsilon();
      }
      if (kind == 3) a[badcell](0, 0) = std::numeric_limits<T>::infinity();
      if (kind == 4)
        du[snap::IPR * 2 + badcell] = std::numeric_limits<T>::infinity();
      bool ok = snap::ForwardSweep(a, b, c, delta, du, 1., 0, 1, 0, 0, 2, 1,
                                   true, true);
      EXPECT_EQ(ok, kind == 0)
          << "N=" << N << " cell=" << badcell << " kind=" << kind;
      if (ok) {
        EXPECT_TRUE(delta[0].allFinite());
        EXPECT_TRUE(delta[1].allFinite());
      }
    }
  }
}

void retry_case(int scheme, bool stop, bool assembly_failure = false,
                torch::Device device = torch::kCPU,
                bool near_singular = false) {
  using namespace snap;
  auto node = YAML::Load(R"(
geometry:
  type: cartesian
  cells: {nx1: 8, nx2: 4, nx3: 1, nghost: 3}
  bounds: {x1min: 0, x1max: 8, x2min: 0, x2max: 4, x3min: 0, x3max: 1}
dynamics:
  equation-of-state: {type: ideal-gas, gammad: 1.4, weight: 0.029}
  reconstruct:
    vertical: {type: plm}
    horizontal: {type: plm}
  riemann-solver: {type: lmars}
forcing:
  const-gravity: {grav1: -1, gravity-work: face, gravity-work-fixer: false}
integration: {type: rk3, implicit-scheme: 9, cfl: 0.5}
boundary-condition:
  external: {x1-inner: reflecting, x1-outer: reflecting,
             x2-inner: periodic, x2-outer: periodic,
             x3-inner: periodic, x3-outer: periodic}
)");
  node["integration"]["implicit-scheme"] = scheme;
  auto path = std::string("test-lu-retry-") + std::to_string(scheme) +
              (stop ? "-stop.yaml" : "-redo.yaml");
  {
    std::ofstream card(path);
    card << node;
  }
  auto opts = MeshBlockOptionsImpl::from_yaml(path);
  std::remove(path.c_str());
  auto block = MeshBlock(opts);
  block->to(device);
  if (near_singular) block->to(torch::kFloat32);
  auto coord = block->pcoord;
  auto w = torch::zeros(
      {5, coord->options->nc3(), coord->options->nc2(), coord->options->nc1()},
      torch::TensorOptions()
          .dtype(near_singular ? torch::kFloat32 : torch::kFloat64)
          .device(device));
  w[IDN].fill_(1);
  w[IPR].fill_(1.e5);
  Variables vars{{"hydro_w", w}};
  block->initialize(vars);
  auto saved = vars.at("hydro_u").clone();
  block->named_buffers()["u0"].copy_(saved);
  // Simulate an earlier RK stage: retry/stop must restore STEP input, not
  // merely the correction input. The remaining hydro state is finite.
  vars.at("hydro_u")[IPR].mul_(1.01);
  auto du = torch::ones_like(w);
  auto before = du.clone();
  auto gamma =
      torch::full_like(w[IDN], (assembly_failure || near_singular)
                                   ? 1.4
                                   : std::numeric_limits<double>::quiet_NaN());
  auto prim = vars.at("hydro_w").clone();
  if (near_singular)
    prim[IPR].mul_(1. + 0.01 * torch::arange(prim.size(-1), prim.options()));
  auto prim0 = prim.clone();
  if (near_singular) {
    ASSERT_TRUE(torch::isfinite(du).all().item<bool>());
    ASSERT_TRUE(torch::isfinite(prim).all().item<bool>());
    ASSERT_TRUE(torch::isfinite(gamma).all().item<bool>());
    // Finite float32 assembly with a long, positive time step: rejection
    // must come from the relative pivot guard, not a NaN or dt=0 input.
    block->phydro->picorr->forward_masked(du, prim, gamma, 1.e4,
                                          torch::Tensor());
    EXPECT_TRUE(block->phydro->picorr->solve_failed());
  } else {
    block->phydro->picorr->forward(du, prim, gamma, assembly_failure ? 0. : 1.);
  }
  EXPECT_TRUE(torch::equal(du, before));
  EXPECT_TRUE(torch::equal(prim, prim0));
  if (stop) block->pintg->current_redo = block->pintg->options->max_redo();
  block->cycle = 7;
  EXPECT_EQ(block->check_redo(vars), stop ? -1 : 1);
  EXPECT_EQ(block->cycle, 7);
  EXPECT_TRUE(torch::equal(vars.at("hydro_u"), saved));
}

}  // namespace

TEST(lu_failure, float_three) { factor_cases<float, 3>(); }
TEST(lu_failure, float_five) { factor_cases<float, 5>(); }
TEST(lu_failure, double_three) { factor_cases<double, 3>(); }
TEST(lu_failure, double_five) { factor_cases<double, 5>(); }
TEST(lu_failure, current_float_three) { sweep_cases<float, 3>(); }
TEST(lu_failure, current_float_five) { sweep_cases<float, 5>(); }
TEST(lu_failure, current_double_three) { sweep_cases<double, 3>(); }
TEST(lu_failure, current_double_five) { sweep_cases<double, 5>(); }
TEST(lu_failure, partial_retry_restores_step) { retry_case(1, false); }
TEST(lu_failure, full_retry_restores_step) { retry_case(9, false); }
TEST(lu_failure, partial_stop_restores_step) { retry_case(1, true); }
TEST(lu_failure, full_stop_restores_step) { retry_case(9, true); }
TEST(lu_failure, partial_assembly_retry_restores_step) {
  retry_case(1, false, true);
}
TEST(lu_failure, full_assembly_retry_restores_step) {
  retry_case(9, false, true);
}
TEST(lu_failure, partial_assembly_stop_restores_step) {
  retry_case(1, true, true);
}
TEST(lu_failure, full_assembly_stop_restores_step) {
  retry_case(9, true, true);
}

TEST(lu_failure, cuda_partial_assembly_retry_restores_step) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CPU build or no CUDA device";
  retry_case(1, false, true, torch::kCUDA);
}
TEST(lu_failure, cuda_full_assembly_retry_restores_step) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CPU build or no CUDA device";
  retry_case(9, false, true, torch::kCUDA);
}
TEST(lu_failure, cuda_partial_assembly_stop_restores_step) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CPU build or no CUDA device";
  retry_case(1, true, true, torch::kCUDA);
}
TEST(lu_failure, cuda_full_assembly_stop_restores_step) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CPU build or no CUDA device";
  retry_case(9, true, true, torch::kCUDA);
}

TEST(lu_failure, mesh_terminal_failure_restores_all_blocks) {
  using namespace snap;
  auto opts = MeshBlockOptionsImpl::from_yaml("test_mesh_multi_block.yaml");
  opts->coord()->nx1(8);
  opts->coord()->nghost(3);
  opts->layout()->px(2);
  opts->layout()->py(1);
  auto gravity = ConstGravityOptionsImpl::create();
  gravity->grav1(-1.);
  gravity->gravity_work("face");
  gravity->gravity_work_fixer(false);
  opts->hydro()->grav() = gravity;
  auto implicit = ImplicitOptionsImpl::create();
  implicit->scheme(9);
  opts->hydro()->icorr() = implicit;
  auto mo = MeshOptionsImpl::create();
  mo->block(opts);
  mo->blocks_per_process(2);
  auto mesh = Mesh(mo);
  ASSERT_EQ(mesh->blocks.size(), 2);
  MeshVariables vars(2);
  for (int n = 0; n < 2; ++n) {
    auto coord = mesh->blocks[n]->pcoord;
    auto w = torch::zeros({5, coord->options->nc3(), coord->options->nc2(),
                           coord->options->nc1()},
                          torch::kFloat64);
    w[IDN].fill_(1.);
    w[IPR].fill_(1.e5);
    vars[n]["hydro_w"] = w;
  }
  mesh->initialize(vars);
  std::vector<torch::Tensor> saved;
  for (int n = 0; n < 2; ++n) {
    saved.push_back(vars[n].at("hydro_u").clone());
    mesh->blocks[n]->named_buffers()["u0"].copy_(saved.back());
    vars[n].at("hydro_u")[IPR].mul_(1.01);
    mesh->blocks[n]->pintg->current_redo =
        mesh->blocks[n]->pintg->options->max_redo();
  }
  auto prim = vars[1].at("hydro_w").clone();
  auto du = torch::ones_like(prim);
  auto gamma = torch::full_like(prim[IDN], 1.4);
  mesh->blocks[1]->phydro->picorr->forward(du, prim, gamma, 0.);
  mesh->set_cycle(7);
  EXPECT_EQ(mesh->check_redo(vars), -1);
  for (auto const &block : mesh->blocks) EXPECT_EQ(block->cycle, 7);
  for (int n = 0; n < 2; ++n)
    EXPECT_TRUE(torch::equal(vars[n].at("hydro_u"), saved[n]))
        << "local block " << n << " was not restored on terminal failure";
}

TEST(lu_failure, finite_near_singular_forward_masked) {
  retry_case(9, false, false, torch::kCPU, true);
}

TEST(lu_failure, cuda_finite_near_singular_forward_masked) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CPU build or no CUDA device";
  retry_case(9, false, false, torch::kCUDA, true);
}

namespace {
void rest_column_clamp(torch::Dtype dtype) {
  using namespace snap;
  auto opts = MeshBlockOptionsImpl::from_yaml("test_mesh_multi_block.yaml");
  opts->coord()->nx1(8);
  opts->coord()->nx2(2);
  opts->coord()->nx3(1);
  opts->coord()->nghost(3);
  opts->coord()->global_nx1(8);
  opts->coord()->global_nx2(2);
  opts->coord()->global_nx3(1);
  opts->coord()->x2max(1.);
  opts->coord()->x3max(1.);
  opts->layout()->px(1);
  opts->layout()->py(1);
  auto gravity = ConstGravityOptionsImpl::create();
  gravity->grav1(0.);
  gravity->gravity_work("face");
  gravity->gravity_work_fixer(false);
  opts->hydro()->grav() = gravity;
  auto implicit = ImplicitOptionsImpl::create();
  implicit->scheme(9);
  opts->hydro()->icorr() = implicit;
  auto block = MeshBlock(opts);
  block->to(dtype);
  auto coord = block->pcoord;
  auto w = torch::zeros(
      {5, coord->options->nc3(), coord->options->nc2(), coord->options->nc1()},
      dtype);
  w[IDN].fill_(1.);
  w[IPR].fill_(1.);
  Variables vars{{"hydro_w", w}};
  block->initialize(vars);
  auto prim = vars.at("hydro_w").clone();
  auto before = prim.clone();
  auto du = torch::zeros_like(prim);
  auto gamma = torch::full_like(prim[IDN], 1.4);
  for (int step = 0; step < 2; ++step) {
    auto correction = block->phydro->picorr->forward_masked(du, prim, gamma, 1.,
                                                            torch::Tensor());
    ASSERT_FALSE(block->phydro->picorr->solve_failed());
    EXPECT_TRUE(torch::equal(du, torch::zeros_like(du)));
    EXPECT_TRUE(torch::equal(correction, torch::zeros_like(correction)));
    EXPECT_TRUE(torch::equal(prim, before));
    auto residual = block->phydro->picorr->clamp_residual();
    EXPECT_TRUE(torch::isfinite(residual).all().item<bool>());
    EXPECT_EQ(residual.item<double>(), 0.);
  }
}
}  // namespace

TEST(lu_failure, float_rest_column_finite_vicclamp) {
  rest_column_clamp(torch::kFloat32);
}
TEST(lu_failure, double_rest_column_finite_vicclamp) {
  rest_column_clamp(torch::kFloat64);
}
