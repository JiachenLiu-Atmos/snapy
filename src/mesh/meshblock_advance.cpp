// MeshBlock, part BlockAdvance (owner X; ops from R, S, EF, K): the time step
// and one stage. (V3 X5: src/mesh/meshblock.cpp in six parts)

// C/C++
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

// snap
#include <snap/coord/coord_utils.hpp>
#include <snap/coord/gnomonic_equiangle.hpp>
#include <snap/eos/ideal_moist.hpp>
#include <snap/hydro/flux_positivity.hpp>
#include <snap/hydro/gravity_work_radial.hpp>
#include <snap/implicit/implicit_dispatch.hpp>
#include <snap/input/read_restart_file.hpp>
#include <snap/mesh/signal_handler.hpp>
#include <snap/output/output_formats.hpp>
#include <snap/output/restart_metadata.hpp>
#include <snap/utils/log.hpp>

#include "cycle_diagnostics.hpp"
#include "meshblock.hpp"
#include "meshblock_internal.hpp"
#include "stage_executor.hpp"

namespace snap {

static c10::impl::GenericDict stage_forcing_variables(
    MeshBlockImpl const &block, Variables const &vars) {
  c10::impl::GenericDict inputs(c10::StringType::get(), c10::TensorType::get());
  for (auto const &item : block.named_buffers(/*recurse=*/true)) {
    inputs.insert(item.key(), item.value());
  }
  for (auto const &[name, value] : vars) {
    inputs.insert_or_assign(name, value);
  }
  return inputs;
}

static Variables stage_forcing_result(c10::IValue const &result, size_t index) {
  TORCH_CHECK(result.isGenericDict(), "User stage forcing ", index,
              " must return Dict[str, Tensor], but returned ", result.tagKind(),
              ".");
  Variables output;
  for (auto const &item : result.toGenericDict()) {
    TORCH_CHECK(item.key().isString(), "User stage forcing ", index,
                " returned a dictionary with a non-string key.");
    TORCH_CHECK(item.value().isTensor(), "User stage forcing ", index,
                " returned a non-Tensor value for key '",
                item.key().toStringRef(), "'.");
    output.emplace(item.key().toStringRef(), item.value().toTensor());
  }
  return output;
}

double MeshBlockImpl::max_time_step(Variables const &vars) {
  return global_time_step(local_max_time_step(vars));
}

double MeshBlockImpl::global_time_step(double dt_local) const {
  std::vector<at::Tensor> dt_reduce = {
      torch::tensor({dt_local}, torch::dtype(torch::kFloat64))};
  if (_playout->has_process_group())
    _playout->comm->allreduce(dt_reduce, c10d::ReduceOp::MIN);
  auto dt = dt_reduce[0].item<double>();
  return pow(2., -pintg->current_redo) * pintg->options->cfl() * dt;
}

double MeshBlockImpl::local_max_time_step(Variables const &vars) const {
  auto const &w = vars.at("hydro_w");
  double dt_min = 1.e9;

  // hyperbolic hydro time step
  if (vars.count("solid")) {
    dt_min = phydro->max_time_step(w, vars.at("solid"));
  } else {
    dt_min = phydro->max_time_step(w);
  }

  return dt_min;
}

void MeshBlockImpl::forward(Variables &vars, double dt, int stage) {
  // Exchange BEFORE reconstruction consumes the ghosts (same count per step):
  // host-side operator-split source terms applied between forward calls
  // (e.g. condensation on hydro_u) are then synced before use, removing a
  // one-signed seam bias on the cubed sphere.
  _executor->run_stage(vars, dt, stage);  // exchange_ghost_zones, then
                                          // advance_local, or their graph
}

void MeshBlockImpl::advance_local(Variables &vars, double dt, int stage) {
  TORCH_CHECK(stage >= 0 && stage < pintg->stages.size(),
              "Invalid stage: ", stage);
  DeferVaporCheck defer(phydro->peos.get());

  // Publish the stage: the vertical implicit correction is nonlinear in dt and
  // must weight dt INSIDE the solve, so it needs to know which stage it is in.
  phydro->rk_stage = stage;
  auto &freg = phydro->flux_register();

  _state.adopt(vars);  // a rebound hydro_u/scalar_s is copied in
  auto hydro_u = vars.at("hydro_u");
  auto scalar_s =
      vars.count("scalar_s") ? vars.at("scalar_s") : torch::Tensor();

  // -------- (1) save initial state --------
  if (stage == 0) {
    // the register's step sums and monitor start the step (R.rst 3.4),
    // before the split sources' step-start half, whose post_split must
    // survive into the step
    freg.begin_step();
    auto splits = phydro->sources.splits();
    for (auto &s : splits) s->begin_split(hydro_u, dt, freg);
    if (!splits.empty()) {
      apply_boundaries(vars, hydro_u, scalar_s);
      exchange_ghost_zones(vars);
    }
    _state.backup();
    if (auto moist = dynamic_cast<IdealMoistImpl *>(phydro->peos.get()))
      moist->clear_heat_capacity_failure();
    if (phydro->picorr) {
      phydro->picorr->reset_dry_clamp_step();
      phydro->picorr->reset_solve_failure();
    }
    phydro->peos->reset_limiter_marks(hydro_u);
    saturation_failures();  // a failure counted before the step is not its own
    if (phydro->peos->vapor_failures().defined())
      phydro->peos->vapor_failures().zero_();  // nor is a vapor failure
  }

  freg.begin_stage(stage, pintg->stages[stage]);

  // -------- (2) set containers for future results --------
  torch::Tensor fut_hydro_du, fut_scalar_ds;

  // A dry source is inside the RK tendency. Addition carries the ratio at
  // source entry; removal carries the complete source-free stage result so it
  // cannot amplify a ratio that transport or the implicit solve just changed.
  auto carry_dry_source = [&](torch::Tensor const &dry,
                              torch::Tensor const &scalar_base) {
    if (!dry.defined() || !vars.count("scalar_r")) return;
    auto rho_base = pintg->forward(stage, _state.u0()[IDN], hydro_u[IDN],
                                   fut_hydro_du[IDN] - dry);
    // anomaly storage: the stage weights of the two states sum to one, so
    // the total is the combined anomaly plus the reference
    if (phydro->has_reference())
      rho_base = rho_base + phydro->u_ref_like(hydro_u)[0];
    auto s_base = pintg->forward(stage, _state.s0(), scalar_s, scalar_base);
    auto remove_r =
        torch::where(rho_base.unsqueeze(0) == 0., vars.at("scalar_r"),
                     s_base / rho_base.unsqueeze(0));
    auto carry_r =
        torch::where(dry.unsqueeze(0) < 0., remove_r, vars.at("scalar_r"));
    fut_scalar_ds.add_(carry_r * dry.unsqueeze(0));
  };

  // -------- (3) launch all jobs --------
  // (3.A) hydro forward
  fut_hydro_du = phydro->forward(dt, hydro_u, vars);

  // (3.B) scalar forward
  if (pscalar->nvar() > 0) {
    fut_scalar_ds = pscalar->forward(dt, scalar_s, vars);
    auto mass_corr = phydro->implicit_mass_correction();
    if (mass_corr.defined() && mass_corr.numel() > 0 &&
        vars.count("scalar_r")) {
      // r is per dry air: move it with the clamped dry transfer, not total M
      auto M = mass_corr[IVY];  // slot IVX+(dir+1)%3 with dir == 0 (x1 only)
      auto r = vars.at("scalar_r");
      int nc1 = r.size(-1);
      auto r_below = torch::zeros_like(r);
      r_below.slice(-1, 1, nc1) = r.slice(-1, 0, nc1 - 1);
      auto P = torch::where(M > 0., r_below, r) * M;
      // cap each donor at held + inflow below, as in vic_constituent_column
      auto held =
          ((scalar_s + fut_scalar_ds).clamp_min(0.) * pcoord->cell_volume())
              .to(P.options());
      at::native::donor_cap(P.device().type(), P, held, pcoord->il(),
                            pcoord->iu());
      auto P_above = torch::zeros_like(P);
      P_above.slice(-1, 0, nc1 - 1) = P.slice(-1, 1, nc1);
      fut_scalar_ds.add_((P - P_above) / pcoord->cell_volume());
    }
    // Native dry sources follow transport and the implicit tracer transfer.
    carry_dry_source(phydro->forcing_dry_increment(), fut_scalar_ds);
  }

  // (3.C) user stage forcings
  auto apply_extra_forcing = [&](Variables const &extra_forcing,
                                 std::string const &source) {
    // the two keys in map order (hydro_du, then scalar_ds), as added before
    for (auto const &[name, d] : extra_forcing) {
      TORCH_CHECK(name == "hydro_du" || name == "scalar_ds", source,
                  " returned unsupported key '", name,
                  "'. Expected one or both of: hydro_du, scalar_ds.");
      bool scalar = name == "scalar_ds";
      if (scalar) {
        TORCH_CHECK(
            pscalar->nvar() > 0, source,
            " returned scalar_ds, but no scalar variables are configured on "
            "this MeshBlock.");
        TORCH_CHECK(fut_scalar_ds.defined(), source,
                    " returned scalar_ds before native scalar tendencies were "
                    "initialized.");
      }
      auto &target = scalar ? fut_scalar_ds : fut_hydro_du;
      TORCH_CHECK(d.sizes() == target.sizes(), source, " returned ", name,
                  " with shape ", d.sizes(), ", expected ", target.sizes(),
                  ".");
      target.add_(d);
    }
  };

  // User scalar_ds is an independent additive tracer source. Keep the scalar
  // base from before the callbacks so a simultaneous dry removal does not debit
  // that explicitly supplied tracer.
  bool track_user_dry = !user_stage_forcings.empty() && pscalar->nvar() > 0 &&
                        vars.count("scalar_r");
  auto user_dry_before =
      track_user_dry ? fut_hydro_du[IDN].clone() : torch::Tensor();
  auto user_scalar_before =
      track_user_dry ? fut_scalar_ds.clone() : torch::Tensor();
  // A user forcing's species increment comes after the flux positivity
  // limiter and the implicit correction, and nothing else bounds it by what
  // they left: clip it so that u + du stays >= 0 in every species cell the
  // forcing would take below zero. The withheld amount is not applied (the
  // forcing removes less than it asked for there). Where the clip does not
  // bind, du is unchanged bit for bit.
  int nspec = hydro_u.size(0) - ICY;
  bool clip_species = !user_stage_forcings.empty() && nspec > 0 &&
                      phydro->options->eos()->limiter();
  auto species_base = clip_species ? fut_hydro_du.narrow(0, ICY, nspec).clone()
                                   : torch::Tensor();
  if (!user_stage_forcings.empty()) {
    auto inputs = stage_forcing_variables(*this, vars);
    for (size_t i = 0; i < user_stage_forcings.size(); ++i) {
      auto result = user_stage_forcings[i]->forward({inputs, dt, stage});
      apply_extra_forcing(stage_forcing_result(result, i),
                          "User stage forcing " + std::to_string(i));
    }
  }
  if (clip_species) {
    auto du = fut_hydro_du.narrow(0, ICY, nspec);
    auto floor = torch::minimum(species_base, -hydro_u.narrow(0, ICY, nspec));
    // only the binding cells are rewritten (signed zeros stay as they are)
    du.copy_(torch::where(du < floor, floor, du));
  }

  if (user_dry_before.defined()) {
    carry_dry_source(fut_hydro_du[IDN] - user_dry_before, user_scalar_before);
  }

  // the stage total is final: S_d <- w1 S_d + w2 dt T_d
  freg.end_stage(dt);

  // -------- (4) multi-stage averaging --------
  pintg->forward(stage, _state.u0(), hydro_u, fut_hydro_du, hydro_u);
  if (stage + 1 == static_cast<int>(pintg->stages.size()))
    for (auto &s : phydro->sources.splits()) s->finish_split(hydro_u, dt, freg);
  // skipped dry and scalar-free before the last stage: next U->W applies it
  if (hydro_u.size(0) > ICY || pscalar->nvar() > 0 ||
      stage + 1 == static_cast<int>(pintg->stages.size()))
    phydro->peos->apply_conserved_limiter_(hydro_u, /*whole_column=*/true);

  if (pscalar->nvar() > 0) {
    pintg->forward(stage, _state.s0(), scalar_s, fut_scalar_ds, scalar_s);
    set_scalar_primitive(vars, scalar_s, hydro_u, *phydro);
  }

  // -------- (5) update ghost zones --------

  if (vars.count("solid")) {
    pib->fill_cons_solid_(hydro_u, vars.at("solid"),
                          vars.at("fill_solid_hydro_u"));
  }

  // -------- (6) saturation adjustment --------
  if (stage == pintg->stages.size() - 1 && phydro->options->eos()->thermo() &&
      phydro->options->eos()->thermo()->reactions().size() > 0) {
    phydro->peos->apply_conserved_limiter_(hydro_u, /*whole_column=*/true);

    int ny = hydro_u.size(0) - ICY;  // number of species

    // thermodynamics needs totals; only the species rows are written back
    auto ut = phydro->total_u(hydro_u);
    auto ke = phydro->peos->compute("U->K", {ut});
    auto rho = ut[IDN] + ut.narrow(0, ICY, ny).sum(0);
    auto ie = ut[IPR] - ke;

    auto yfrac = hydro_u.narrow(0, ICY, ny) / rho;

    auto pthermo = thermo();
    TORCH_CHECK(pthermo, "saturation adjustment needs hydro.eos.thermo");

    auto sub = part({0, 0, 0}, PartOptions().exterior(false));
    auto sub3 = part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
    pthermo->forward(rho.index(sub3), ie.index(sub3), yfrac.index(sub),
                     /*warm_start=*/true);

    hydro_u.narrow(0, ICY, ny) = yfrac * rho;
  }

  // the global phase of a bare block (a Mesh runs it over its blocks)
  if (phydro->paam->closes() && !defer_axial_closure)
    apply_axial_closure(vars, dt, stage,
                        global_phase_sum({axial_closure_sums()}, *_playout));

  // Physical ghosts must include the final-stage species adjustment.
  apply_boundaries(vars, hydro_u, scalar_s);
}

torch::Tensor MeshBlockImpl::axial_closure_sums() const {
  auto const &s = phydro->axial_closure_sums();  // device, FP64 (MPS: FP32)
  auto wide = pcoord->x1v.is_mps() ? torch::kFloat32 : torch::kFloat64;
  return s.defined() ? s.clone()
                     : torch::zeros({2}, pcoord->x1v.options().dtype(wide));
}

void MeshBlockImpl::apply_axial_closure(Variables &vars, double dt, int stage,
                                        torch::Tensor const &global) {
  // lane A's fixer (A.rst 3.3), with this stage's weight; R and I stay on
  // the device (X4): a mask replaces the host branch
  AxialClosureFixer(phydro->paam.get())
      .apply(vars.at("hydro_u"), global, dt, pintg->stages[stage].wght2(),
             phydro->flux_register());
}

}  // namespace snap
