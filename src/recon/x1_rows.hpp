#pragma once

#include <ATen/TensorIterator.h>
#include <torch/torch.h>

#include <snap/utils/tensor_cast_cache.hpp>
#include <utility>

#include "recon_point.h"

// Every x1 weight, built on the host in double from the faces: the
// reconstruction rows, conversion order 4 (L), the key-4 wall ghosts, the
// wb_ref4 face density and the stretched six-face reference rows. Consumers
// cast to their device and dtype and rebuild when X1RowsKey changes.

namespace snap {

//! weights of the n cell averages between faces z[0..n] that give, at x, the
//! value of the polynomial whose primitive interpolates them (n <= 5). The
//! reconstruction rows and the wb_ref4 face density are this one formula
void face_value_row(double const* z, int n, double x, double* w);

std::pair<torch::Tensor, torch::Tensor> weno5_x1_coeffs(
    torch::Tensor const& x1f);

std::pair<torch::Tensor, torch::Tensor> cp5_x1_coeffs(torch::Tensor const& x1f);

void x1_reset_tables(torch::nn::Module& m, torch::Tensor& cm_x1,
                     torch::Tensor& cp_x1, int ncoeff);

template <typename Self, typename Builder>
void x1_set_coeffs(Self& self, torch::Tensor const& x1f, Builder build) {
  auto rows = build(x1f);

  auto opts = self.cm.options();
  auto cm_rows = rows.first.to(opts).contiguous();
  auto cp_rows = rows.second.to(opts).contiguous();
  if (self.named_buffers(/*recurse=*/false).contains("cm_x1")) {
    self.cm_x1.set_data(cm_rows);
    self.cp_x1.set_data(cp_rows);
  } else {
    self.cm_x1 = self.register_buffer("cm_x1", cm_rows);
    self.cp_x1 = self.register_buffer("cp_x1", cp_rows);
  }

  self.cm_x1_cast = {};
  self.cp_x1_cast = {};
}

inline at::TensorIterator x1_uniform_iterator(torch::Tensor const& w, int dim,
                                              torch::Tensor const& out) {
  std::vector<int64_t> squash_dim = {0};
  if (w.device().is_cuda()) squash_dim.push_back(dim);
  return at::TensorIteratorConfig()
      .resize_outputs(false)
      .check_all_same_dtype(true)
      .declare_static_shape(out.sizes(), squash_dim)
      .add_output(out)
      .add_input(w)
      .build();
}

torch::Tensor x1_coeffs_row_view(torch::Tensor const& c,
                                 torch::Tensor const& out, int dim);

at::TensorIterator x1_nu_iterator(torch::Tensor const& w, int dim,
                                  torch::Tensor const& out,
                                  torch::Tensor const& c);

void x1_coeffs_check_rows(at::TensorIterator const& iter,
                          torch::Tensor const& c, int dim, int ncoeff,
                          char const* what);

template <typename Call>
void x1_forward_nu(torch::Tensor const& w, int dim, torch::Tensor const& out,
                   torch::Tensor const& src, TensorCastCache& cache,
                   char const* what, Call call) {
  TORCH_CHECK(dim == out.dim() - 1, "[", what,
              "] route A covers the x1 dimension only, so a module holding "
              "per-face x1 coefficients must be given the last dimension of "
              "a (nvar, nc3, nc2, nc1) state; got dim ",
              dim, " of a ", out.dim(), "-dimensional output (note Sec. 4.7)");

  TORCH_CHECK(out.size(dim) == src.size(0), "[", what,
              "] per-face x1 coefficients are installed but dimension ", dim,
              " has ", out.size(dim), " outputs while the table has ",
              src.size(0),
              " rows; route A covers the x1 dimension only, so `recon23` must "
              "not be given a face vector (note Sec. 4.7)");

  auto c = cache.get(src, w);
  auto iter = x1_nu_iterator(w, dim, out, c);
  call(iter, c);
}

//! the one cache key of the x1 weights: x1f (storage and version) and the
//! consumer's device and dtype. A consumer rebuilds when its key changes
struct X1RowsKey {
  void const* faces = nullptr;
  int64_t version = -1;
  c10::Device device = c10::kCPU;
  c10::ScalarType dtype = c10::ScalarType::Undefined;

  X1RowsKey() = default;
  X1RowsKey(torch::Tensor const& x1f, torch::TensorOptions const& options)
      : faces(x1f.defined() ? x1f.data_ptr() : nullptr),
        version(x1f.defined() ? x1f._version() : -1),
        device(options.device()),
        dtype(options.dtype().toScalarType()) {}

  bool operator==(X1RowsKey const& o) const {
    return faces == o.faces && version == o.version && device == o.device &&
           dtype == o.dtype;
  }
  bool operator!=(X1RowsKey const& o) const { return !(*this == o); }
};

//! conversion order 4: L a = h^2/24 q''(x_i), q the quadratic whose averages
//! over cells i-1, i, i+1 are a, as {kp, km} on cells 1..n-2 of n = nc1
std::pair<torch::Tensor, torch::Tensor> conv4_rows(torch::Tensor const& x1f,
                                                   int n);

//! conversion order 4 at a reflecting x1 wall, (2, 4, ng, 4): owned
//! averages to ghosts (lane-c-notes.md)
torch::Tensor wall4_rows(torch::Tensor const& x1f, int is, int iu, int ng);

//! #289, conversion order 4: the face density of the cell reference at faces
//! is..iu+1, (nown + 1, 4) cell indices and weights, the derivative at the
//! face of the quartic primitive through five faces (four cells). The window
//! stays inside the owned cells at a clamped physical wall
struct FaceDensityRows {
  int is = 0, iu = -1, nc1 = 0;
  //! false: a clamped wall with fewer than four owned cells; nothing changes
  bool usable = false;
  bool clamp_in = false, clamp_out = false;
  torch::Tensor idx, wt;
  //! true for the cells that count for the resolution flag
  torch::Tensor counted;
  //! the x1 rows' cache key at construction
  X1RowsKey key;
};
FaceDensityRows face_density_rows(torch::Tensor const& x1f, int is, int iu,
                                  bool clamp_in, bool clamp_out,
                                  torch::TensorOptions const& options);

//! stretched x1 reference: per cell its first face and the six-face weights
//! of its average, (nc1, 7); a start below zero has no row
torch::Tensor six_face_rows(torch::Tensor const& dx1f, int iu, bool phys_in,
                            bool phys_out, bool wall_clamp);

}  // namespace snap
