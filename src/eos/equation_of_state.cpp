// C/C++
#include <algorithm>

// yaml
#include <yaml-cpp/yaml.h>

// torch
#include <ATen/TensorIterator.h>

// kintera
#include <kintera/species.hpp>

// snap
#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>
#include <snap/hydro/flux_positivity.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/input/check_keys.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/utils/log.hpp>

#include "aneos.hpp"
#include "eos_dispatch.hpp"
#include "equation_of_state.hpp"
#include "ideal_gas.hpp"
#include "ideal_moist.hpp"
#include "moist_mixture.hpp"
#include "shallow_water.hpp"

namespace snap {

EquationOfStateOptions EquationOfStateOptionsImpl::from_yaml(
    std::string const& filename, bool verbose) {
  auto config = YAML::LoadFile(filename);
  auto op = EquationOfStateOptionsImpl::create();

  if (!config["dynamics"]) return op;
  TORCH_CHECK(config["dynamics"].IsMap() || config["dynamics"].IsNull(),
              "EquationOfStateOptions: dynamics must be a map.");
  if (config["dynamics"].IsNull()) return op;
  if (!config["dynamics"]["equation-of-state"]) return op;

  auto node = config["dynamics"]["equation-of-state"];

  TORCH_CHECK(
      node.IsMap() || node.IsNull(),
      "EquationOfStateOptions: dynamics/equation-of-state must be a map.");
  if (node.IsNull()) return op;
  check_keys(node, "dynamics/equation-of-state",
             {"type", "gammad", "weight", "density-floor", "pressure-floor",
              "temperature-floor", "limiter", "eos-file", "verbose"},
             "kintera", {"max-iter", "ftol", "uv-solver"});

  op->verbose() = node["verbose"].as<bool>(verbose);

  op->type() = node["type"].as<std::string>("moist-mixture");
  if (op->verbose()) {
    SINFO(EquationOfStateOptions) << "EOS type = " << op->type() << std::endl;
  }

  op->gammad() = node["gammad"].as<double>(1.4);
  op->weight() = node["weight"].as<double>(29.e-3);

  op->density_floor() = node["density-floor"].as<double>(1.e-6);
  if (op->verbose()) {
    SINFO(EquationOfStateOptions)
        << "density floor = " << op->density_floor() << std::endl;
  }

  op->pressure_floor() = node["pressure-floor"].as<double>(1.e-3);
  op->temperature_floor() = node["temperature-floor"].as<double>(20.);

  op->limiter() = node["limiter"].as<bool>(false);
  if (op->verbose()) {
    SINFO(EquationOfStateOptions)
        << "limiter = " << (op->limiter() ? "true" : "false") << std::endl;
  }

  op->eos_file() = node["eos-file"].as<std::string>("");
  if (op->verbose() && !op->eos_file().empty()) {
    SINFO(EquationOfStateOptions)
        << "eos file = " << op->eos_file() << std::endl;
  }

  op->thermo() = kintera::ThermoOptionsImpl::from_yaml(filename, op->verbose());

  if (op->thermo()) {
    TORCH_CHECK(
        NMASS == 0 || (op->thermo()->vapor_ids().size() +
                           op->thermo()->cloud_ids().size() ==
                       1 + NMASS),
        "Athena++ style indexing is enabled (NMASS > 0), but the number of "
        "vapor and cloud species in the thermodynamics options does not match "
        "the expected number of vapor + cloud species = ",
        1 + NMASS);
  }

  return op;
}

EquationOfStateImpl::EquationOfStateImpl(EquationOfStateOptions const& options_,
                                         torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  TORCH_CHECK(phydro, "[EquationOfState] Parent module is null.");
  cache_cloud_parents_();
}

void EquationOfStateImpl::cache_cloud_parents_() {
  if (!options->thermo()) return;

  auto const& thermo = options->thermo();
  int const nvapor = thermo->vapor_ids().size();
  int const ncloud = thermo->cloud_ids().size();
  cloud_parent_cache_.assign(ncloud, {});

  auto const nucleation = thermo->nucleation();
  if (!nucleation) return;

  // names()/mu() are this thermo's own copy of the species table, in
  // species() order (vapors, then clouds), taken when its card was parsed.
  // kintera's global table (species_names, species_weights) refills when a
  // later card declares other species (kintera #121), so vapor_ids/cloud_ids
  // must not be resolved through it here (issue #234). A thermo not built
  // from a card has no copy yet; populate_thermo takes one from the table as
  // it is now, as kintera does when its own modules are built.
  kintera::populate_thermo(thermo);
  auto const& names = thermo->names();
  auto const& mu = thermo->mu();
  TORCH_CHECK(names.size() == nvapor + ncloud && mu.size() == names.size(),
              "[EquationOfState] thermo carries ", names.size(),
              " species names and ", mu.size(), " molar masses for ",
              nvapor + ncloud, " species.");

  for (int j = 0; j < ncloud; ++j) {
    double parent_mass = 0.;
    for (auto const& reaction : nucleation->reactions()) {
      if (!reaction.products().count(names[nvapor + j])) continue;

      for (auto const& [parent, coefficient] : reaction.reactants()) {
        // vapor slot 0 is the dry gas and is never a parent
        auto vapor_it =
            std::find(names.begin() + 1, names.begin() + nvapor, parent);
        if (vapor_it == names.begin() + nvapor) continue;

        int vapor_index = std::distance(names.begin(), vapor_it);
        double mass = coefficient * mu[vapor_index];
        cloud_parent_cache_[j].emplace_back(ICY + vapor_index - 1, mass);
        parent_mass += mass;
      }
      break;
    }

    if (parent_mass <= 0.) {
      cloud_parent_cache_[j].clear();
      continue;
    }
    for (auto& parent : cloud_parent_cache_[j]) {
      parent.second /= parent_mass;
    }
  }
}

torch::Tensor EquationOfStateImpl::internal_energy_offset(
    torch::Tensor hydro_like) const {
  return torch::zeros_like(hydro_like[IDN]);
}

torch::Tensor EquationOfStateImpl::specific_heat_cv(torch::Tensor prim,
                                                    torch::Tensor temp) {
  return torch::zeros_like(temp);
}

torch::Tensor EquationOfStateImpl::compute(
    std::string ab, std::vector<torch::Tensor> const& args) {
  TORCH_CHECK(false, "[EquationOfState] compute() is not implemented.",
              "Please use this method in a derived class.");
}

/*torch::Tensor EquationOfStateImpl::get_buffer(std::string) const {
  TORCH_CHECK(false, "[EquationOfState] get_buffer() is not implemented.",
              "Please use this method in a derived class.");
}*/

torch::Tensor EquationOfStateImpl::forward(torch::Tensor cons,
                                           torch::optional<torch::Tensor> out) {
  auto prim = out.value_or(torch::empty_like(cons));
  return compute("U->W", {cons, prim});
}

void EquationOfStateImpl::apply_conserved_limiter_(torch::Tensor const& cons,
                                                   bool whole_column) {
  auto pmb = phydro->pmb;
  auto pcoord = pmb->pcoord;

  if (!options->limiter()) return;  // no limiter
  // every call marks a repaired interior, so MeshBlock::check_redo redoes it
  auto interior = pmb->part({0, 0, 0}, PartOptions().exterior(false));
  bool mark = limiter_marks_.defined();
  auto nan = torch::isnan(cons);
  if (mark) push_mark_(1, nan.index(interior).any());
  cons.masked_fill_(nan, 0.);
  auto dens_energy = [&] {  // density only where the EOS has no energy row
    auto u = cons.index(interior);
    return nvar() > IPR ? torch::stack({u[IDN], u[IPR]}) : u[IDN].clone();
  };
  auto before = mark ? dens_energy() : torch::Tensor();
  cons[IDN].clamp_min_(options->density_floor());

  // for (int i = ICY; i < ICY + nvapor; ++i)
  //   cons.index(interior)[i] = pull_neighbors3(cons.index(interior)[i]);
  //  batched
  // cons.index(interior).narrow(0, ICY, nvapor) =
  //    pull_neighbors4(cons.index(interior).narrow(0, ICY, nvapor));

  int ny = 0;
  int nvapor = 0;
  int ncloud = 0;
  if (options->thermo()) {
    nvapor = options->thermo()->vapor_ids().size() - 1;
    ncloud = options->thermo()->cloud_ids().size();
    ny = nvapor + ncloud;
  }

  if (nvar() > IPR) {
    auto mom = cons.narrow(0, IVX, 3).clone();
    coord_vec_raise_(mom, pcoord->cosine_cell_kj);
    auto rho = cons[IDN] + cons.narrow(0, ICY, ny).sum(0);
    auto ke = 0.5 * (mom * cons.narrow(0, IVX, 3)).sum(0) / rho;
    auto min_temp = options->temperature_floor() * torch::ones_like(ke);
    auto min_ie = compute("UT->I", {cons, min_temp});
    cons[IPR].clamp_min_(ke + min_ie);
  }
  if (mark) push_mark_(0, dens_energy().ne(before).any());

  if (options->thermo() && ny > 0) {
    auto nghost = pcoord->options->nghost();
    auto species = [&] { return cons.index(interior).narrow(0, ICY, ny); };
    auto species_before = mark ? species().clone() : torch::Tensor();

    // A negative condensate value means the tracer flux over-drained the cell;
    // the mass is in a neighbour, not missing. Zeroing it (`clamp_min_(0.)`)
    // invented mass one-way -- measured at 102% of the total-mass drift in
    // silicate cloud runs. Instead, borrow the deficit from the parent vapor in
    // the SAME cell: exactly conservative in mass, energy and elements, and
    // per-cell, so ghost copies receive the identical repair and ranks stay in
    // sync. A cell whose vapor cannot cover the deficit goes vapor-negative and
    // is handled by the columnar fix_vapor below. The phase shift is ~1% of the
    // local value and the saturation adjustment re-equilibrates it at the end
    // of the same cycle.
    bool parentless = false;
    for (int j = 0; j < ncloud; ++j) {
      int slot = ICY + nvapor + j;
      auto c = cons[slot];
      auto const& parents = cloud_parent_cache_[j];
      if (parents.empty()) {
        // Clouds not produced by nucleation have no parent-vapor metadata;
        // they are repaired along the column below.
        parentless = true;
        continue;
      }

      auto deficit = c.clamp_max(0.);
      for (auto const& [parent_slot, mass_fraction] : parents) {
        cons[parent_slot] += deficit * mass_fraction;
      }
      c.clamp_min_(0.);  // condensate to exactly zero
    }

    // cell volumes, so a column repair conserves mass, not density (#241)
    auto vol = pcoord->cell_volume().unsqueeze(0).contiguous().index(interior);

    // Keep the two repairs separate: clouds tolerate an unrepairable column
    // and clamp it; vapor must throw before a gathered column is copied back.
    auto repair_column = [&](torch::Tensor field, bool require_success) {
      int count = field.size(0);
      auto major = cons.index(interior)[IDN].unsqueeze(0);
      auto layout = pmb->get_layout();
      bool split =
          count > 0 && whole_column && layout && layout->options->pz() > 1;
      auto column = split ? layout->gather_x1(torch::cat(
                                {field, major, vol.expand_as(major)}))
                          : torch::Tensor();
      auto repaired = split ? column.narrow(0, 0, count) : field;
      auto cmajor = split ? column.narrow(0, count, 1) : major;
      auto cvol = split ? column.narrow(0, count + 1, 1) : vol;
      auto iter = at::TensorIteratorConfig()
                      .resize_outputs(false)
                      .declare_static_shape(repaired.sizes(),
                                            /*squash_dim=*/repaired.dim() - 1)
                      .add_output(repaired)
                      .add_owned_input(cmajor.expand_as(repaired))
                      .add_owned_input(cvol.expand_as(repaired))
                      .build();
      // deferred: the count waits for check_redo (on CUDA, on the device)
      bool defer = require_success && defer_vapor_check_;
      if (defer && (!vapor_failures_.defined() ||
                    vapor_failures_.device() != cons.device()))
        vapor_failures_ = torch::zeros({}, cons.options().dtype(torch::kInt));
      int err = at::native::call_fix_vapor(
          cons.device().type(), iter, require_success,
          defer ? vapor_failures_ : torch::Tensor());
      if (defer && err > 0) vapor_failures_ += err;  // the CPU returns it
      TORCH_CHECK(!require_success || defer || err == 0,
                  "[EquationOfState] apply_conserved_limiter_: "
                  "Failed to fix vapor mass fractions.");
      if (split) {
        int nx1 = field.size(-1);
        int rz = std::get<2>(layout->loc_of(layout->options->rank()));
        field.copy_(repaired.narrow(-1, rz * nx1, nx1));
      }
    };

    if (parentless) {
      repair_column(cons.index(interior).narrow(0, ICY + nvapor, ncloud),
                    false);
      cons.narrow(0, ICY + nvapor, ncloud).clamp_min_(0.);
    }
    repair_column(cons.index(interior).narrow(0, ICY, nvapor), true);
    // mark on the deficit the repair covered, not on how far it moved the
    // donors: a deficit below kSpeciesRepairMarkUlp of the cell's total gas
    // density is repaired without a redo; kinetics leaves ~1e-304 in a
    // cloud-free cell that no smaller dt removes (#256)
    if (mark) {
      auto rho = cons.index(interior)[IDN] + species_before.sum(0);
      auto tol = species_repair_mark(cons.scalar_type()) * rho.abs();
      push_mark_(0, (species_before < -tol).any());
    }
  }
}

void EquationOfStateImpl::apply_primitive_limiter_(torch::Tensor const& prim) {
  if (!options->limiter()) return;  // no limiter
  auto nan = torch::isnan(prim);
  if (limiter_marks_.defined()) {  // floors on a prim are floor_hit's to judge
    auto interior = phydro->pmb->part({0, 0, 0}, PartOptions().exterior(false));
    push_mark_(1, nan.index(interior).any());
  }
  prim.masked_fill_(nan, 0.);
  prim[IDN].clamp_min_(options->density_floor());

  if (options->thermo()) {
    int ny = options->thermo()->vapor_ids().size() +
             options->thermo()->cloud_ids().size() - 1;
    if (limiter_marks_.defined()) {  // floor_hit sees no species row
      auto interior =
          phydro->pmb->part({0, 0, 0}, PartOptions().exterior(false));
      // mass fractions: the same round-off bound as the conserved repair
      push_mark_(0, (prim.index(interior).narrow(0, ICY, ny) <
                     -positivity_roundoff(prim.scalar_type()))
                        .any());
    }
    prim.narrow(0, ICY, ny).clamp_min_(0.);
  }

  prim[IPR].clamp_min_(options->pressure_floor());
}

void EquationOfStateImpl::reset_limiter_marks(torch::Tensor const& like) {
  limiter_marks_ = torch::zeros({2}, like.options().dtype(torch::kBool));
  pending_marks_[0].clear();
  pending_marks_[1].clear();
}

EquationOfState EquationOfStateImpl::create(EquationOfStateOptions const& opts,
                                            torch::nn::Module* p,
                                            std::string const& name) {
  TORCH_CHECK(p, "[EquationOfState] Parent module pointer is null.");
  TORCH_CHECK(opts, "[EquationOfState] Options pointer is null.");

  if (opts->type() == "ideal-gas") {
    return p->register_module(name, IdealGas(opts, p));
  } else if (opts->type() == "ideal-moist") {
    return p->register_module(name, IdealMoist(opts, p));
  } else if (opts->type() == "moist-mixture") {
    return p->register_module(name, MoistMixture(opts, p));
  } else if (opts->type() == "aneos") {
    return p->register_module(name, ANEOS(opts, p));
  } else if (opts->type() == "shallow-water") {
    return p->register_module(name, ShallowWater(opts, p));
  } else {
    TORCH_CHECK(false, "EquationOfState: Unknown type: ", opts->type());
  }
}

}  // namespace snap
