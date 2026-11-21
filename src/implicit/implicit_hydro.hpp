#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// snap
#include <snap/forcing/forcing.hpp>
#include <snap/hydro/flux_register.hpp>
#include <snap/hydro/gravity_work.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {

//! The VIC matrices: the full 5x5 blocks, or the partial 3x3 blocks of rows
//! and columns {rho, m1, E} (implicit-scheme 9 and 1); the value is the size.
enum class VicMatrix : int { partial = 3, full = 5 };

struct ImplicitOptionsImpl {
  static std::shared_ptr<ImplicitOptionsImpl> create() {
    return std::make_shared<ImplicitOptionsImpl>();
  }
  static std::shared_ptr<ImplicitOptionsImpl> from_yaml(
      std::string const& filename, bool verbose = false);
  static std::shared_ptr<ImplicitOptionsImpl> from_yaml(const YAML::Node& node);

  ImplicitOptionsImpl() = default;
  void report(std::ostream& os) const {
    os << "-- implicit hydro options --\n";
    os << "* type = " << type() << "\n"
       << "* precision = " << precision() << "\n"
       << "* fallback_fp64 = " << fallback_fp64() << "\n"
       << "* scheme = " << scheme() << "\n"
       << "* advection_cfl = " << advection_cfl() << "\n"
       << "* shear_cfl = " << shear_cfl() << "\n";
  }

  //! the VIC matrix: scheme 9 (bit 3 set) is the full one
  VicMatrix matrix() const {
    return (scheme() >> 3) & 1 ? VicMatrix::full : VicMatrix::partial;
  }
  int size() const { return static_cast<int>(matrix()); }

  std::string type() const;
  ADD_ARG(int, scheme) = 0;
  ADD_ARG(std::string, precision) = "auto";
  ADD_ARG(bool, fallback_fp64) = true;
  ADD_ARG(double, advection_cfl) = 1.0;
  ADD_ARG(double, shear_cfl) = 0.0;
};
using ImplicitOptions = std::shared_ptr<ImplicitOptionsImpl>;

class HydroImpl;

class ImplicitHydroImpl : public torch::nn::Cloneable<ImplicitHydroImpl> {
 public:
  //! Create and register a ImplicitHydro module
  /*!
   * This function registers the created module as a submodule
   * of the given parent module `p`.
   *
   * \param[in] opts      options for creating the `ImplicitHydro` module
   * \param[in] p         parent module for registering the created module
   * \param[in] name      name for the created module
   * \return              created `ImplicitHydro` module
   */
  static std::shared_ptr<ImplicitHydroImpl> create(
      ImplicitOptions const& opts, torch::nn::Module* p,
      std::string const& name = "icorr");

  //! options with which this `ImplicitHydro` was constructed
  ImplicitOptions options;

  //! non-owning pointer to parent
  HydroImpl const* phydro = nullptr;

  //! Constructor to initialize the layer
  ImplicitHydroImpl() : options(ImplicitOptionsImpl::create()) {}
  explicit ImplicitHydroImpl(ImplicitOptions const& options,
                             torch::nn::Module* p = nullptr);
  void reset() override;

  //! corrector for the implicit hydro
  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor gamma,
                        double dt);
  //! V3 I-4: with reg and the step's post_dt (> 0), the face transfers are
  //! posted to reg in phase implicit (add "vic.mass_dry", "vic.mass_species",
  //! "vic.momentum_t"; keep "vic.mass_total"; the amounts "vic.cell",
  //! "vic.mass_clamped" and, full matrix, "vic.momentum_t.top"), each as
  //! M_f / (A_f post_dt), and the gravity-work remainder reads the kept
  //! "vic.mass_total" (I.rst section 4)
  //! V3 I-6: with retry ([nc3, nc2] bool) defined, the predicated FP64
  //! stage: only those columns are solved, every other one is left as it
  //! came in (and posts -0)
  torch::Tensor forward_masked(torch::Tensor du, torch::Tensor w,
                               torch::Tensor gamma, double dt,
                               torch::Tensor solid, FluxRegister* reg = nullptr,
                               double post_dt = 0.,
                               torch::Tensor const& retry = {});

  //! I-6: the columns the last FP32 (or only) stage failed, [nc3, nc2] bool
  //! on the device: the FP64 stage's predicate
  torch::Tensor stage_failed() const { return _stage_failed; }
  //! after the FP64 stage: the FP32 stage's correction where no column was
  //! retried, the FP64 stage's where one was, in the FP32 stage's dtype
  torch::Tensor merge_fp64(torch::Tensor const& retry);
  //! fp64_fallbacks counts the retried columns (I.rst section 10(c))
  void count_fp64_columns(torch::Tensor const& retry);
  torch::Tensor correction() const { return _corr; }
  torch::Tensor mass_correction() const { return _mass_corr; }

  //! running max relative break of the VIC telescoping identity (clamp meter)
  torch::Tensor clamp_residual() const { return _clamp_residual; }

  //! 1 if the dry-gas clamp emptied a cell since the last reset, else 0
  torch::Tensor dry_clamp_step() const { return _dry_clamp_step; }
  void reset_dry_clamp_step() { _dry_clamp_step.zero_(); }

  //! V3 I-5: each column's VicStatusBit [nc3, nc2] (int32), set by the
  //! column's kernel thread and by the result check, kept over the step's
  //! stages until reset_status() (step start)
  torch::Tensor col_status() const;
  //! I-6: whether a column's last solve failed (the FP64 retry failed, or a
  //! failure no FP64 retry followed), OR-reduced, 0-d bool, on the device
  torch::Tensor status() const;
  void reset_status();
  //! on a redo only (the host already synchronised): the failed columns
  void report_failure(std::ostream& os) const;

  //! status() read back: the FP32 fallback's host read (hydro.cpp, until I-6)
  bool solve_failed() const { return status().item<bool>(); }
  void reset_solve_failure() { reset_status(); }

  //! status() as a 0-d tensor (1 or 0) with `opts`, without a host read: the
  //! redo's slot 5 (meshblock.cpp) until lane X reads retry_flags
  torch::Tensor solve_failed_tensor(torch::TensorOptions const& opts) const {
    return status().to(opts);
  }

 private:
  void ensure_workspace(torch::Tensor const& w);
  torch::Tensor _fp64_fallbacks;
  torch::Tensor _status;        // [nc3, nc2] int32: the step's VicStatusBit
  torch::Tensor _stage_status;  // [nc3, nc2] the stage's, solve dtype (kernel)
  torch::Tensor _stage_failed;  // [nc3, nc2] bool: the FP64 stage's predicate
  //! the FP64 stage's scratch, swapped in for that stage (I-6)
  torch::Tensor _a64, _b64, _c64, _delta64, _du064, _corr64, _mass_corr64,
      _stage_status64;
  void swap_scratch_();

  torch::Tensor _a, _b, _c, _delta, _du0, _corr, _mass_corr, _clamp_residual,
      _dry_clamp_step;

  //! the gravity work's grid-only coefficients: W's face weights and the
  //! potentials of the remainder, K's rows and factors (V3 I-3)
  GravityWork _gw;

  //! the post-solve gravity-work remainder, one call (V3 I-3): the work of the
  //! projection and the availability clamp, then the corrected potential
  //! energy's remainder g1 sigma^2 s[du - du0] beyond what the matrix booked;
  //! added to du[IPR] in that order
  void book_remainder_(torch::Tensor du, int wsize, torch::Tensor const& mask,
                       bool no_solid, bool radial_exact, double grav1,
                       torch::Tensor const& rx_mass0,
                       torch::Tensor const& rx_fluid, int nx1,
                       torch::Tensor const& moved_total);
  //! I-4: the posts, the kept total and the amounts, after the stage's
  //! failure mask
  void post_(FluxRegister& reg, torch::Tensor const& du,
             torch::Tensor const& drop, torch::Tensor const& kdt,
             torch::Tensor const& ptan, torch::Tensor const& energy_solved,
             torch::Tensor const& volume, torch::Tensor const& kept_before);
};
TORCH_MODULE(ImplicitHydro);

}  // namespace snap

#undef ADD_ARG
